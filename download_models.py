from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Nicemleme yapılandırması
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Modeli 4-bit nicemleme ile yükle
    bnb_4bit_use_double_quant=True,  # Çift nicemleme kullan
    bnb_4bit_quant_type="nf4",  # Nicemleme türü (NormalFloat4)
    bnb_4bit_compute_dtype="float16",  # Hesaplamalar için float16 kullan
)

# Model ve tokenizer'ı yükle
model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=quantization_config,
)

# Modeli kullanma
input_text = "Merhaba, nasılsın?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))