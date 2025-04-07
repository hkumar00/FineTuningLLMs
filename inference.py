from peft import PeftModel

ADAPTER_PATH = "./phi2-qlora-alpaca"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # Avoid mixed precision issues
    low_cpu_mem_usage=True,
    device_map=None,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model = model.merge_and_unload()

model.eval()

# ========== INFERENCE ==========
prompt = "Explain how transformers work in AI/ML"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=500)
    print(tokenizer.decode(output[0], skip_special_tokens=True))