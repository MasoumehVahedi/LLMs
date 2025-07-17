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


