## Fine-Tuning a Large Language Model – *Product Pricer*

This project fine-tunes an LLM so it can **predict a product’s price from its text description**.

---

### 1️⃣  Get (or create) a dataset
| Where to look | Good when… |
|---------------|------------|
| **Your own tickets / emails / CMS pages** | You need the model to speak your company’s jargon. |
| **Kaggle**    | You just want a quick public demo. |
| **Hugging Face Datasets** | One-liner `load_dataset()` and you’re ready. |
| **Synthetic data** (use GPT-4 to generate) | Real data is scarce or sensitive. |
| **Label vendors (e.g. Scale AI)** | You need thousands of human-checked rows fast. |

---

### 2️⃣  “Data curation”  

*Data curation* = **investigate → clean → balance → save**.  
Think of it as building the *nutrition label* for your model’s diet.

| Step | What to do | Why |
|------|------------|-----|
| **Investigate** | Count rows, look at min/median/max length, null fields. | Know what you’re feeding the model. |
| **Parse** | Turn raw rows into *prompt → target* pairs.<br>Remove HTML, normalise units. | Consistent input format. |
| **Visualise** | Plot length histograms and label distribution. | Outliers jump out visually. |
| **Assess quality** | Manually read 50 random samples. | Quick sanity check before hours of training. |
| **Curate** | • Deduplicate near-identical rows<br>• Drop rows longer than the context window<br>• Down-sample over-represented classes | Prevent bias & wasted tokens. |
| **Save** | Write `train.jsonl` and `eval.jsonl` with keys `prompt`, `response`. | Trainer libraries expect this. |

*Curation includes preprocessing, exploring, balancing, and documenting the data.*

---

### 3️⃣  “Scrub” the data (optimisation pass)

* Remove PII or brand names unless the task requires them.  
* Make labels uniform:  
  `"$12.99"`, `"USD 12.99"` → `"USD 12.99"`.  
* Enforce one style: consistent casing, punctuation, date formats.  
* Split very long docs into smaller paragraphs so every sample fits the
  model’s max tokens.  
* For tricky tasks, prepend 2–3 **few-shot examples** to each prompt.  
* Run GPT-4 or Claude‐2 as a **proof-reader** to catch grammar errors or
  hallucinated targets.

_Re-plot after each scrub.  Distributions should look smoother, with fewer crazy outliers._

---

### ✅  What you end up with
A **clean, balanced, well-documented dataset** that is:

* small enough to fine-tune cheaply (LoRA or full-finetune)  
* large enough to teach the new behaviour  
* trusted enough to deploy without nasty surprises

Fine-tuning will converge faster, cost less, and generalise better than if
you dumped raw, messy data into the trainer. 

