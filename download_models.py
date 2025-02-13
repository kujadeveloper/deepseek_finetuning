import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import BitsAndBytesConfig

# Nicemleme yapılandırması oluştur
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit nicemleme etkinleştir
    bnb_4bit_use_double_quant=True,  # Çift nicemleme kullan
    bnb_4bit_quant_type="nf4",  # NormalFloat4 kullan
    bnb_4bit_compute_dtype=torch.float16,  # Hesaplamalar için float16
)

# Model konfigürasyonu ile birlikte yükle
model_name = "deepseek-ai/DeepSeek-R1"

# Tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Model yükle (quantization ayarı AutoConfig ile)
config = AutoConfig.from_pretrained(model_name)
config.quantization_config = quantization_config

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    trust_remote_code=True
)

# Modeli test et
input_text = "Merhaba, nasılsın?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
