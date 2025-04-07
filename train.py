import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

# ========== DEVICE SETUP ==========
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

# ========== LOAD MODEL & TOKENIZER ==========
#model_name = "mistralai/Mistral-7B-v0.1"
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # Avoid mixed precision issues
    low_cpu_mem_usage=True,
    device_map=None,  # Use "auto" for MPS compatibility
)
base_model.to(device)

# ========== PREPARE MODEL FOR PEFT ==========
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, peft_config)

# Optional: Avoid MPS gradient issues
model.gradient_checkpointing_disable()
for name, module in model.named_modules():
    if "lora" in name.lower():
        module.to(torch.float32)

# ========== LOAD ALPACA DATASET ==========
# Manually load a local version or a small sample
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1%]")  # ~500 examples for testing

# ========== TOKENIZATION ==========
def format_text(text):
    prompt = text["instruction"]
    input_text = text["input"]
    answer = text["output"]

    if input_text:
        full_prompt = f"Instruction: {prompt}\nInput: {input_text}\n\nResponse:"
    else:
        full_prompt = f"Instruction: {prompt}\n\nResponse:"

    tokenized = tokenizer(
            full_prompt + " " + answer,
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


dataset = dataset.map(format_text, batched=False)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ========== DATALOADER COLLATOR ==========
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ========== TRAINING ARGUMENTS ==========
training_args = TrainingArguments(
    output_dir="./mistral-alpaca-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    learning_rate=2e-4,
    bf16=False,
    fp16=False,
    do_train=True,
    report_to="none",
    no_cuda=True,  # Required for CPU/MPS
)

# ========== TRAINER ==========
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #tokenizer=tokenizer,
    data_collator=collator
)

# ========== START TRAINING ==========
trainer.train()


model.save_pretrained("./phi2-qlora-alpaca")
tokenizer.save_pretrained("./phi2-qlora-alpaca")