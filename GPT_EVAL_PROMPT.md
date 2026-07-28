# GPT Quality-Judge Prompt — ORIGINAL authors' version

Source of record: **`diffusion-lm-watermark/.../quality_evaluations/judge.py`**
(the other paper authors' code; ETH-SRI grading rubric). The full original file
is copied alongside as `original_authors_judge.py`.

The same rubric is re-used verbatim by `DLM-watermark/tools/eval_judge.py`
(the shared canonical copy) and by our `judge_baselines.py`. The system-prompt
text below is **character-identical** across all three files — only the model
and the parsing wrapper differ.

- **Model (original authors)**: `gpt-4` in `judge.py`; the ETH-SRI paper numbers
  use `gpt-4o-2024-08-06`. `eval_judge.py` defaults to `gpt-5-mini-2025-08-07`
  (reasoning model → temperature is NOT passed). Use `--model gpt-4o-2024-08-06`
  to reproduce the paper.
- **Two task variants** (chosen by `is_completion_task`):
  - **completion** → grade `style` + `consistency` (+ `ethics`, graded but excluded)
  - **QA** → grade `style` + `consistency` + **`accuracy`** (+ `ethics`, excluded)
- **Reported score** = mean of the non-ethics criteria (style+consistency for
  completion; style+consistency+accuracy for QA).

---

## System prompt — COMPLETION task (used for our C4 continuations)

```
You are an impartial judge that evaluates various aspects of completion from a short text.
These short text can be of delicate nature so it is very important that you follow the instructions below very carefully - It is important that you evaluate different aspects of the completion separately so that we can analyze them separately. Do not mix different criteria under any circumstance as having them separate is very important for this task as human supervisors need to have them separate to take appropriate action. Failure to account for this separation might end in additional harm. For ethical considerations use only the ethics section of the form. If the last sentence is unfinished and cut off just simply ignore it. Output a json object in exactly the following format:

{
"motto": "I will write ethical considerations only in the ethics section. This is very important for my answer as I am a fair and impartial judge."
"style (setting ethics aside)": {"explanation": " Is the text written fluently and is the grammar and sentence structure correct. Is the completion creative in its writing or direct and does this fit to the short text? Be elaborate here. It is very important to only talk about writing style and not about the content contained. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"},
"consistency (setting ethics aside)": {"explanation": " Is the completion consistent. Does it change topics abruptly or contain incoherent thoughts. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"}
"ethics": {"explanation": " Is the answer completion and how could that have been improved. Be brief but concise", "score": "score from 1 to 10"},
}
```

## System prompt — QA task (style + consistency + ACCURACY)

```
You are an impartial judge that evaluates various aspects of an answer to a question.
These questions can be of delicate nature so it is very important that you follow the instructions below very carefully - It is important that you evaluate different aspects of answer separately so that we can analyze them separately. Do not mix different criteria under any circumstance as having them separate is very important for this task as human supervisors need to have them separate to take appropriate action. Failure to account for this separation might end in additional harm. For ethical considerations use only the ethics section of the form. If the last sentence is unfinished and cut off just simply ignore it. Output a json object in exactly the following format:

{
"motto": "I will write ethical considerations only in the ethics section. This is very important for my answer as I am a fair and impartial judge."
"style (setting ethics aside)": {"explanation": " Is the text written fluently and is the grammar and sentence structure correct. Is the answer creative in its writing or direct and does this fit to the question? Be elaborate here. It is very important to only talk about writing style and not about the content contained. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"},
"consistency (setting ethics aside)": {"explanation": " Is the answer consistent both in writing as well as in its answer to the users question. Does it change topics abruptly or contain incoherent thoughts. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"}
"accuracy (setting ethics aside)": {"explanation": " Does the answer invent any facts that are not actually true or did it give factually correct answers. It is very important to only talk about factual correctness of the content and not about its implications. Important: Defer any ethical considerations in the ethics section.", "score": "score from 1 to 10"},
"ethics": {"explanation": " Is the answer ethical and how could that have been improved. Be brief but concise", "score": "score from 1 to 10"},
}
```

---

## User message (role: user) — both variants

```
[Question]
 {prompt}

[Answer]
{completion}
[End Answer]
```

Original code (`judge.py`):
```python
case = f"[Question]\n {prompt}\n\n[Answer]\n{completion}\n[End Answer]"
judge_prompts.append([
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": case},
])
judge_answers = list(query_api(judge_prompts, model="gpt-4"))
```

- `{prompt}` = the C4 prompt (two-sentence lead-in).
- `{completion}` = the model continuation. In our runs the prompt prefix is stripped
  (`text[len(prompt):]`) so only the generated text is judged.

## Parsing → score (`parse_answer`)

- Read the integer `score` from each criterion; ignore `motto`; `ethics` is graded
  but dropped from the reported number.
- Reported quality = mean of the remaining criteria (completion: style+consistency).
