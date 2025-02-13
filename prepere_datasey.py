from datasets import load_dataset

# Veri setini yükle
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Veri setini tokenize et
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)