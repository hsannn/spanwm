"""Retrain IE's entropy tagger for OUR setting (Llama-3.2-3B on C4).

Why this is necessary: IE ships five taggers (Carol0110/IE, tau in
{0.3,0.6,0.9,1.2,1.5}) so nothing needs training in principle -- but their
labels are StarCoder-15.5B next-token entropies on MBPP CODE. Measured on our
data (2400 positions of Llama-3.2-3B C4 continuations) all five score AUROC
0.45-0.50, i.e. exactly chance, and accuracy far below the majority baseline.
Their tau grid is also mis-scaled: at tau=0.3, 90% of our positions are
"high entropy", because natural text is far less predictable than code.

So we redo the distillation faithfully but in-domain:
  features : SimCSE (sup-simcse-roberta-base) last hidden state at the last
             non-pad position of the decoded prefix   [IE's exact recipe]
  labels   : y=1 iff TRUE next-token entropy (Llama-3.2-3B) < tau, i.e.
             1 = LOW entropy -- IE's polarity (paper 4.2: "0 for high-entropy
             tokens and 1 for low-entropy tokens"), which the inference code
             relies on: calculate_entropy returns softmax[:,1] = P(low) and
             score_sequence scores tokens with P(low) < 0.5
  tau grid : our measured entropy percentiles (P25/P50/P75 = 0.94/2.21/3.48),
             the same values the SWEET sweep uses, so the two are comparable
  model    : IE's exact Classifier MLP (768-512-256-128-64-32-2), BCE, AdamW
             lr 1e-4, wd 2e-5, batch 32

Training documents come from C4 rows 1000+ (our evaluation uses rows 0-199),
so there is no overlap with anything reported.

    python train_ie_tagger.py --n_docs 300 --max_pos 100
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

SIMCSE = "princeton-nlp/sup-simcse-roberta-base"
TAUS = (0.9, 2.2, 3.5)


class Classifier(nn.Module):
    """IE's tagger architecture, unchanged."""

    def __init__(self, input_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return self.output(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="dataset/c4/processed_c4.json")
    ap.add_argument("--start_row", type=int, default=1000)
    ap.add_argument("--n_docs", type=int, default=300)
    ap.add_argument("--max_pos", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--outdir", default="watermark/ie/model")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    ltok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(dev).eval()
    stok = AutoTokenizer.from_pretrained(SIMCSE)
    smod = AutoModel.from_pretrained(SIMCSE).to(dev).eval()

    rows = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i < args.start_row:
                continue
            if len(rows) >= args.n_docs:
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"training docs: {len(rows)} (C4 rows {args.start_row}+)", flush=True)

    feats, ents = [], []
    for k, r in enumerate(rows):
        text = (r.get("prompt", "") + r.get("natural_text", "")).strip()
        ids = ltok(text, return_tensors="pt", add_special_tokens=False,
                   truncation=True, max_length=256)["input_ids"].to(dev)
        if ids.shape[1] < 12:
            continue
        with torch.no_grad():
            probs = torch.softmax(lm(ids).logits.float(), dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)[0].cpu().numpy()
        n = min(ids.shape[1] - 1, args.max_pos)
        prefixes = [ltok.decode(ids[0, :i + 1], skip_special_tokens=True)
                    for i in range(n)]
        for j in range(0, len(prefixes), 64):
            enc = stok(prefixes[j:j + 64], return_tensors="pt", padding=True,
                       truncation=True, max_length=512,
                       add_special_tokens=False).to(dev)
            with torch.no_grad():
                hs = smod(**enc).last_hidden_state
            last = enc["attention_mask"].sum(1) - 1
            feats.append(hs[torch.arange(hs.size(0)), last].float().cpu())
        ents.extend(ent[:n].tolist())
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(rows)} docs, {len(ents)} positions", flush=True)

    X = torch.cat(feats)
    y_ent = torch.tensor(ents, dtype=torch.float32)
    print(f"dataset: {X.shape[0]} positions, entropy mean={y_ent.mean():.3f} "
          f"median={y_ent.median():.3f}", flush=True)
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(42))
    X, y_ent = X[perm], y_ent[perm]
    ntr = int(0.9 * X.shape[0])

    for tau in TAUS:
        # IE polarity: 1 = low entropy (below tau), 0 = high entropy
        y = (y_ent < tau).long()
        Xtr, ytr, Xte, yte = X[:ntr].to(dev), y[:ntr].to(dev), X[ntr:].to(dev), y[ntr:]
        clf = Classifier().to(dev)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-4, weight_decay=2e-5)
        lossf = nn.CrossEntropyLoss()
        best = 0.0
        for ep in range(args.epochs):
            clf.train()
            idx = torch.randperm(Xtr.shape[0], device=dev)
            for i in range(0, Xtr.shape[0] - 31, 32):
                b = idx[i:i + 32]
                loss = lossf(clf(Xtr[b]), ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
                clf.eval()
                with torch.no_grad():
                    pr = torch.softmax(clf(Xte), dim=1)[:, 1].cpu().numpy()
                acc = ((pr > 0.5).astype(int) == yte.numpy()).mean()
                auc = roc_auc_score(yte.numpy(), pr) if yte.float().std() > 0 else float("nan")
                print(f"  tau={tau} ep{ep + 1:3d}: acc={acc:.3f} AUROC={auc:.3f}",
                      flush=True)
                best = max(best, auc)
        path = f"{args.outdir}/entropy_tagger_{str(tau).replace('.', '_')}.pt"
        torch.save(clf.state_dict(), path)
        base = max(float(yte.float().mean()), 1 - float(yte.float().mean()))
        print(f"tau={tau}: saved -> {path}  (majority baseline {base:.3f}, "
              f"best AUROC {best:.3f})", flush=True)
    print("IE TAGGER TRAINING DONE")


if __name__ == "__main__":
    main()
