## LLaMA 3.1 Architecture

LLaMA 3.1 is built from 32 stacked “decoder” modules, each containing:  
- **Self-Attention layers** for contextual token mixing  
- **Multi-Layer Perceptron (MLP) layers** for nonlinear transformations   
- **SiLU activations** and **LayerNorm** to stabilize and scale training  

Together, these layers hold billions of parameters and form the core of the model’s representation power.


## LoRA Fine-Tuning Steps

1. **Freeze Original Weights**  
   Don’t update any of the model’s billions of parameters during optimization.  

2. **Select Target Modules**  
   Choose a handful of layers (e.g. specific attention or MLP blocks) that you’ll adapt—these are your “target modules.”  

3. **Create Low-Rank Adapter Matrices**  
   For each target module, instantiate two small matrices (LoRA_A and LoRA_B) with far fewer parameters than the original layer.  

4. **Inject & Train Adapters**  
   Apply the adapter matrices into the target modules and optimize only those matrices - and these get trained. 

5. **Use Dual Matrices per Module**  
   Each adapter is factored into an “A” and a “B” matrix—together they approximate the weight updates you want, keeping overall memory use minimal.  

### 📝 Summary

LoRA lets you fine-tune massive models by freezing all original weights, picking only key layers to adapt, and inserting lightweight, low-rank adapters (LoRA_A and LoRA_B) into each. By training only these small matrices, you drastically cut memory and compute costs while still steering the model toward your task.  

### Three Essential LoRA Hyperparameters

1. **Rank (r)**  
   – Number of dimensions in each adapter matrix.  
   **Rule of thumb:** start at 8, then double to 16, 32… until you see diminishing returns.  

2. **Alpha (α)**  
   – A scalar multiplier on the adapter update:  
   \[
     \Delta W = \alpha \cdot A \cdot B
   \]  
   where \(A\) and \(B\) are the low-rank adapter matrices.  
   **Rule of thumb:** set \(\alpha = 2 \times r\). Larger \(\alpha\) → stronger effect.  

3. **Target Modules**  
   – Which layers to adapt in the model.  
   **Common choice:** focus on self-attention heads. You can tailor this (e.g. final MLP layers) if your task demands a very different output style or format.  


## Quantization: Why QLoRA?

Even with LoRA’s adapter trick, the base model (e.g. LLaMA-3.1 8B) still requires ~32 GB of 32-bit floating-point precision memory—far more than most single-GPU setups allow. QLoRA solves this by **quantizing** the frozen base weights to lower precision, while still training only the small LoRA adapters:

- **Problem:**  
  - 32-bit floating-point precision storage of 8 B parameters → 8 B × 32 bits = 32 GB  
  - T4/RTX-class GPUs often only have 16 GB or less → base model doesn’t even fit.  

- **Idea:**  
  - **Keep the same number of weights** but drastically **reduce their precision** (e.g. to 8-bit or even 4-bit floats).  
  - Quantized weights “click” between fewer discrete levels, yet preserve most of the model’s expressiveness.  

- **Key Benefits:**  
  1. **Memory footprint** drops from 32 GB → ~4 GB (8-bit) or ~2 GB (4-bit).  
  2. **Performance impact** is surprisingly small—models retain the bulk of their FP32 (or 32-bit floating-point precision) accuracy.  
  3. **LoRA adapters** remain full-precision, so fine-tuning quality is unchanged.  


###  QLoRA Workflow Summary

1. **Quantize** the base model’s frozen weights to 8-bit (or 4-bit) floats.  
2. **Select target modules** and **inject** full-precision LoRA_A/LoRA_B adapters.  
3. **Train** only those adapters on your task (the quantized base stays fixed).  
4. **Deploy** a lightweight, high-performance fine-tuned model that fits on a single GPU.  




