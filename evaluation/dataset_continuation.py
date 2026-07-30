"""Continuation-protocol loaders for the scale-up baseline grid.

These two classes are copied VERBATIM (logic-identical, docstrings aside) from
`spanwm_embed_v8.py` on origin/main, which is where the collaborator defined
them for the SpanWM v8 runs. They live here so `baseline_embed.py` and
`measure_entropy.py` select exactly the same 200 prompts as those runs without
importing from an entry script — and, deliberately, without editing
`evaluation/dataset.py`, which is the collaborator's file.

Protocol, for the record:
    CNN/DailyMail  prompt = the article's first 30 words, natural_text = the
                   rest; articles of <= 30 words are skipped.
    WMT16          prompt = the 'en' sentence; sentences under 10 words are
                   skipped (too little context to draft from). No natural_text.

Both take `lines[:max_samples]` — the FIRST N rows in file order, no shuffling,
the same selection convention as C4Dataset.
"""

import json

from evaluation.dataset import BaseDataset


class WMT16ENDataset(BaseDataset):
    """Continuation on the English side: prompt = 'en' sentence."""

    MIN_WORDS = 10

    def __init__(self, data_source, max_samples=200):
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        with open(self.data_source) as f:
            for line in f:
                if len(self.prompts) >= self.max_samples:
                    break
                en = json.loads(line)["en"]
                if len(en.split()) >= self.MIN_WORDS:
                    self.prompts.append(en)


class CNNArticleDataset(BaseDataset):
    """Continuation over the article body (no summarization instruction)."""

    PROMPT_WORDS = 30

    def __init__(self, data_source, max_samples=200):
        super().__init__(max_samples)
        self.data_source = data_source
        self.load_data()

    def load_data(self):
        with open(self.data_source) as f:
            for line in f:
                if len(self.prompts) >= self.max_samples:
                    break
                words = json.loads(line)["article"].split()
                if len(words) <= self.PROMPT_WORDS:
                    continue
                self.prompts.append(" ".join(words[:self.PROMPT_WORDS]))
                self.natural_texts.append(" ".join(words[self.PROMPT_WORDS:]))
