from transformers import pipeline

# Modeli yükle
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-deepseek-v3-lite")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-deepseek-v3-lite")

# Metin üretme pipeline'ını oluştur
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Metin üret
prompt = "Merhaba, nasılsın?"
output = text_generator(prompt, max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])