# FineTuningLLMs
Collection of scripts to perform finetuning on consumer grade hardware or google colab free tier


# 🔧 LLM Fine-Tuning on Consumer Hardware (CPU / MPS / Free Colab)

This repository provides minimal and **resource-friendly** code to fine-tune large language models (LLMs) using **LoRA** on **custom instruction-style datasets** (like Alpaca) using either:

- ✅ CPU-only or Apple M1/M2 (via MPS)
- ✅ Free Colab GPUs (no paid subscription required)

Our approach allows **modular plug-and-play fine-tuning** across different models and datasets.

---

## 📌 Key Strategies for Consumer-Grade Fine-Tuning

| Parameter                         | Why We Use It                                                                 |
|----------------------------------|-------------------------------------------------------------------------------|
| `LoRA` via `peft`                | Trains only a small set of parameters → saves memory                         |
| `batch_size=1`                   | Prevents OOM errors on CPU / MPS / small GPUs                                |
| `gradient_accumulation_steps=4`  | Simulates larger batch size with less memory                                 |
| `learning_rate=2e-4`             | Empirically effective for small models + LoRA                                |
| `num_train_epochs=1-3`           | Quick convergence for small datasets; can be increased for better results    |
| `fp16=False` on CPU/MPS          | Mixed precision not stable without CUDA                                      |
| `save_total_limit=2`             | Prevents disk overflow from multiple checkpoints                             |
| `alpaca-style datasets`          | Easy to adapt for any task format (instruction + input → response)           |

---

## 🛠️ General Setup

```bash
pip install -r requirements.txt
```
---

🧩 Fine-Tuning Workflow Summary
A typical fine-tuning script in this repo follows these steps:

Load the base model and tokenizer
Use AutoModelForCausalLM and AutoTokenizer from Hugging Face.

Configure LoRA with LoraConfig
Define the low-rank adaptation parameters like r, alpha, and dropout.

Format your dataset
Convert each sample into a consistent Instruction, Input, and Response format (e.g., Alpaca-style).

Fine-tune using SFTTrainer
Pass in your model, dataset, tokenizer (optional), and training arguments.

Save checkpoints or merge LoRA adapter
Save fine-tuned weights and optionally merge LoRA into the base model for standalone inference.



