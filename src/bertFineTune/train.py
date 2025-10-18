from transformers import TrainingArguments, Trainer
from tqdm import tqdm


def train_model(model, tokenizer, train_dataset, test_dataset, compute_metrics):
    training_args = TrainingArguments(
        output_dir="outputs/models/bert_finetuned_openalex",
        num_train_epochs=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1.3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir="./logs",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    tqdm.write("🚀 Starting fine-tuning...")
    trainer.train()
    tqdm.write("✅ Training completed!")

    return trainer
