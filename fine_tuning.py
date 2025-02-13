from transformers import Trainer, TrainingArguments

# Eğitim ayarlarını belirle
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer'ı oluştur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Modeli eğit
trainer.train()

model.save_pretrained("./fine-tuned-deepseek-v3-lite")
tokenizer.save_pretrained("./fine-tuned-deepseek-v3-lite")
