# LLM Robustness under Noisy Instruction Data

A small experiment to see how well small language models hold up when part of their training data is deliberately corrupted.

---

## What's this about

I wanted to see what happens when you fine-tune a small LLM on a mix of correct and corrupted instruction-output pairs. Do the models still learn something useful? How much does the noise hurt?

---

## Dataset

Combined Alpaca (~52k samples) and Databricks Dolly-15k, cleaned up (removed URLs, non-ASCII characters, very short samples), ending up with ~67k samples total.

Split: 80% train, 10% validation, 10% test.

---

## What I corrupted

Applied three types of corruption to the training outputs and mixed them 50/50 with clean data:

- **Character reversal** – `"hello"` → `"olleh"`
- **Word reversal** – `"I like cats"` → `"cats like I"`
- **Counterfactual** – used Gemini 1.5 Flash to generate a wrong but believable answer

---

## Models

Fine-tuned three small models using LoRA (`peft` library):

- `sshleifer/tiny-gpt2`
- `distilgpt2`
- `facebook/opt-125m`

Trained on Colab, so kept things light — `max_steps=50`, `batch_size=1`, `max_length=256`.

---

## Evaluation

Evaluated on 100 test samples per model:

- Semantic similarity (cosine similarity via `all-mpnet-base-v2`)
- BLEU
- ROUGE-L

---

## How to run

```bash
pip install transformers datasets peft accelerate sentence-transformers evaluate google-generativeai rouge_score
```

Run `implementation_final_iisc.ipynb` cell by cell. You'll need a Gemini API key for the counterfactual generation step.

---

Shreya Kale | M.Tech AI/ML | NIT Calicut
