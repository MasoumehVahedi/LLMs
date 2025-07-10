## 5-Step Strategy: From Business Problem to Production LLM

> A repeatable playbook for applying LLMs to real-world tasks.

1. **Understand**  
   - **What?** Deeply capture the business requirements:  
     - Success criteria (model-centric vs business-centric metrics)  
     - Non-functional constraints (cost, latency, scalability)  
   - **Why?** Ensures you measure the right KPIs (training & validation loss, average price error, “good” estimates).

2. **Prepare**  
   - **Research**  
     - Baseline (non-LLM) solutions  
     - Candidate LLMs (context window, cost, license, benchmarks)  
   - **Curate your data**  
     - Clean, preprocess, dedupe  
     - Split into train / validation / test sets  

3. **Select**  
   - **Pick the model(s)** to experiment with (frontier vs open-source).  
   - **Train & validate** on your curated data.  
   - **Hold out** a final test split for unbiased evaluation and hyperparameter tuning.

4. **Customize**  
   - **Prompting** (few-shot, chain-of-thought, function-calling) for quick, low-cost gains.  
   - **RAG** (retrieve + generate) to tap large knowledge bases with minimal fine-tuning.  
   - **Fine-tuning** for deep domain expertise, nuance, style transfer, and faster inference.

   > **Note:** prompting & RAG are _inference-time_ techniques, while fine-tuning is a _training-time_ investment.

5. **Productionize**  
   - **Deploy** into a robust pipeline (monitoring, versioning, latency SLAs).  
   - **Guard against “catastrophic forgetting”** by scheduling periodic re-training & data refresh.  
   - **Report** business metrics (e.g. % of estimates within \$X or Y%) on live traffic.

---

> **Why follow these steps?**  
> This structured journey—from understanding the problem to productionizing your LLM—keeps you aligned with business goals, prevents wasted effort, and makes it easy to compare techniques (prompting, RAG, fine-tuning) in the contexts where they shine.  


## Evaluation Metrics

We’ll track both **model-centric** (technical) and **business-centric** (outcome) metrics:

| **Model-centric** (Training & Debug)      | **Business-centric** (User Impact)         |
|-------------------------------------------|--------------------------------------------|
| **Training loss**<br>– how well you fit the training set | **Average price error**<br>– mean absolute difference between predicted vs actual prices |
| **Validation loss**<br>– generalization on held-out data | **% price difference**<br>– `(|predicted – actual| ÷ actual) × 100%`|
| **Root Mean Squared Log Error (RMSLE)**<br>– penalizes large under-/over-predictions | **“Good” estimate rate**<br>– share of predictions within \$40 **or** within 20% of the true price |

> **Why both?**  
> - **Model-centric** metrics (losses, RMSLE) let you debug and compare architectures or hyperparameters during training.  
> - **Business-centric** metrics (error rates, “hit” percentages) show real-world impact and the fraction of “acceptable” price estimates your users (or stakeholders) care about most.

You will watch your training & validation loss drop epoch-by-epoch. Once fine-tuned, you will compare frontier vs open-source LLMs by their business-centric scores to pick the model that truly delivers value.  

