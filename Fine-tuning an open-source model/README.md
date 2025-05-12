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

