## Three Stages of Fine‑Tuning a Frontier Model

1. **Prepare & Upload Data**  
   • Convert our examples into OpenAI’s JSONL format (one record per line).  
   • Each record must have a `"prompt"` and `"completion"` field.  
   • Upload the JSONL file to OpenAI to kick off a fine‑tune job.

2. **Train & Monitor**  
   • Start the fine‑tuning job on our uploaded dataset.  
   • Watch the training loss and validation loss—both should steadily decrease.  
   • If validation loss stalls or rises, consider adjusting learning rate, batch size, or data quality.

3. **Evaluate & Iterate**  
   • Once training finishes, run our model on held‑out test examples.  
   • Compare predictions vs. ground truth, gather metrics (e.g. RMSE, accuracy).  
   • Tweak prompts, hyperparameters, or dataset—and repeat the cycle.


## How Many Examples for Fine‑Tuning?

- **Recommendation:** OpenAI suggests using **50–100 examples** for a fine‑tune—just enough to teach the model our specific style or edge‑case behaviors.  
- **Why not thousands?** Models like GPT‑4 have already seen billions of tokens; we only need a handful of examples to steer tone, fix quirks, or boost accuracy on our niche task.  
- **Our choice:** We will use **500 examples**—more than the minimum, but still small enough that training is fast and cheap.  


## Weights & Biases Integration

**What is W&B?**  
Weights & Biases (W&B) is a free service for visualizing and tracking our ML experiments—loss curves, metrics, hyperparameters, and more.

**Why integrate with OpenAI fine‑tuning?**  
- Automatically log training & validation loss per epoch  
- Record all hyperparameters (learning rate, batch size, epochs)  
- Monitor job status and metadata in real time  

This makes it far easier to spot overfitting, compare runs, and share results with our team.

**Setup Steps**

1. **Create a W&B account**  
   - Go to [https://wandb.ai](https://wandb.ai)  
   - Sign up or log in  
   - Click your Avatar → **Settings**, then copy our **API Key**

2. **Connect W&B to OpenAI**  
   - Visit [https://platform.openai.com/account/organization](https://platform.openai.com/account/organization)
   - click the ⚙️ (Settings) icon in the top right → Organization → General  
   - Scroll down to the **Integrations** → Weights and Biases section  
   - Paste your W&B API Key under **Weights & Biases**

3. **Enable in code**  
   ```python
   from finetuner import FineTuner

   # Pass our W&B project name into the constructor
   ft = FineTuner(wandb_project="my-wandb-project")

   # Launch your fine‑tune as usual
   job_id = ft.create_fine_tune_job(
       train_file_id, valid_file_id,
       base_model="gpt-4o-mini",
       n_epochs=1,
       suffix="pricer"
   )

