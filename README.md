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
## 🔄 Fine-Tuning Workflow Summary

This section outlines the standard pattern followed in all training scripts within this repo. It ensures modularity, clarity, and compatibility with consumer-grade hardware setups.

### 🔧 Step-by-step Flow:

1. **Load the base model and tokenizer**  
   Use `AutoModelForCausalLM` and `AutoTokenizer` from Hugging Face.  
   Set `torch_dtype=torch.float32` and `low_cpu_mem_usage=True` to reduce memory usage.

2. **Configure LoRA with `LoraConfig`**  
   Leverage `peft` to define low-rank adaptation:
   - `r=8` (rank)
   - `lora_alpha=16` (scaling factor)
   - `lora_dropout=0.1`
   - `bias="none"`
   - `task_type=TaskType.CAUSAL_LM`

3. **Format your dataset**  
   Each record must have:
   - `instruction` (what the model should do)
   - `input` (optional context)
   - `output` (the expected response)

   Format the prompt in Alpaca style:
     Instruction: {instruction} Input: {input}
     Response: {output}

5. **Fine-tune using `SFTTrainer`**  
A lightweight wrapper from `trl` simplifies the process.  
Pass in:
- `model` (with LoRA applied)
- `train_dataset` (after tokenization)
- `TrainingArguments`
- `data_collator` (disable MLM)

5. **Save or merge the LoRA adapter**  
- Use `save_steps` and `save_total_limit` to control checkpointing.
- Optionally merge the LoRA adapter into the base model after training using PEFT utilities.

---


## 🧪 Fine-Tuning `phi-2` on Alpaca (LoRA, CPU/MPS-Friendly)

📁 **Folder:** `./experiments/phi2-alpaca-lora`

### ⚙️ Parameters

| Component             | Setting                                |
|-----------------------|----------------------------------------|
| **Base Model**        | `microsoft/phi-2`                      |
| **Dataset**           | `tatsu-lab/alpaca` (1% sample)         |
| **LoRA Rank (r)**     | 8                                      |
| **LoRA Alpha**        | 16                                     |
| **Dropout**           | 0.1                                    |
| **Max Length**        | 512 tokens                             |
| **Device**            | CPU / MPS (Apple Silicon)              |
| **Batch Size**        | 1                                      |
| **Accumulation**      | 4                                      |
| **Epochs**            | 1                                      |
| **Learning Rate**     | `2e-4`                                 |

### 🧠 Why These Settings?

- `r=8`, `alpha=16` → Good balance between performance and efficiency for LoRA on small devices  
- `gradient_accumulation_steps=4` → Effective batch size = 4  
- `max_length=512` → Shorter sequences enable faster training  
- `output_dir=./mistral-alpaca-lora` → Easy to organize multiple experiments  

---

### 🧾 `format_text()` Function

We use Alpaca-style formatting for supervised fine-tuning:

```python
def format_text(text):
    if text["input"]:
        full_prompt = f"Instruction: {text['instruction']}\nInput: {text['input']}\n\nResponse:"
    else:
        full_prompt = f"Instruction: {text['instruction']}\n\nResponse:"
    tokenized = tokenizer(full_prompt + " " + text["output"], ...)

```

