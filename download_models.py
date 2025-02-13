from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ve tokenizer'ı yükle
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)