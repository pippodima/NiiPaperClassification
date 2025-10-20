from prepare_data import prepare_datasets, prepare_small_datasets
from tokenizer_utils import tokenize_and_encode_labels
from model import get_model, compute_metrics
from train import train_model
from plot_metrics import plot_training_logs


N_EPOCHS = 3


def main():
    model_name = "bert-base-uncased"

    print("📂 Preparing datasets...")
    train_dataset, test_dataset, df = prepare_small_datasets()

    print("🔠 Tokenizing and encoding labels...")
    train_dataset, test_dataset, tokenizer, le = tokenize_and_encode_labels(
        train_dataset, test_dataset, df, model_name=model_name
    )

    print("🧠 Loading model...")
    model = get_model(model_name, num_labels=len(le.classes_))

    print("🏋️ Training model...")
    trainer = train_model(model, tokenizer, train_dataset, test_dataset, compute_metrics, epochs=N_EPOCHS)

    print("💾 Saving model and tokenizer...")
    model.save_pretrained("bert_finetuned_openalex")
    tokenizer.save_pretrained("bert_finetuned_openalex")

    print("🎯 Final Evaluation Metrics:")
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
