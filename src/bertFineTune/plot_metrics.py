import matplotlib.pyplot as plt
import json
import os
import numpy as np


def plot_training_logs(log_dir="./logs", save_path="outputs/plots/bert_training_metrics.png"):
    log_file = os.path.join(log_dir, "trainer_state.json")
    if not os.path.exists(log_file):
        print("⚠️ No log file found in", log_dir)
        return

    with open(log_file, "r") as f:
        data = json.load(f)

    if "log_history" not in data:
        print("⚠️ No metrics found in trainer_state.json")
        return

    logs = data["log_history"]

    # Extract metrics
    epochs = []
    train_loss, eval_loss, acc, f1, lr, prec, rec = [], [], [], [], [], [], []

    for log in logs:
        if "epoch" in log:
            epochs.append(log["epoch"])
            train_loss.append(log.get("loss", np.nan))
            eval_loss.append(log.get("eval_loss", np.nan))
            acc.append(log.get("eval_accuracy", np.nan))
            f1.append(log.get("eval_f1", np.nan))
            lr.append(log.get("learning_rate", np.nan))
            prec.append(log.get("eval_precision", np.nan))
            rec.append(log.get("eval_recall", np.nan))

    # Convert to numpy for easier handling
    epochs = np.array(epochs)
    train_loss = np.array(train_loss)
    eval_loss = np.array(eval_loss)
    acc = np.array(acc)
    f1 = np.array(f1)
    lr = np.array(lr)

    # Create 3 subplots: Loss, Accuracy/F1, Learning Rate
    plt.figure(figsize=(12, 10))

    # ---- 1. Loss ----
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss, marker="o", label="Training Loss")
    plt.plot(epochs, eval_loss, marker="o", label="Eval Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # ---- 2. Accuracy & F1 ----
    plt.subplot(3, 1, 2)
    plt.plot(epochs, acc, marker="o", label="Eval Accuracy")
    plt.plot(epochs, f1, marker="o", label="Eval F1")
    plt.plot(epochs, prec, marker='o', label="Eval Precision")
    plt.plot(epochs, rec, marker='o', label="Eval Recall")
    plt.title("Evaluation Metrics per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # ---- 3. Learning Rate ----
    plt.subplot(3, 1, 3)
    plt.plot(epochs, lr, marker="o", color="purple", label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

    # ---- Print useful stats ----
    print("\n📊 Training Summary:")
    if not np.isnan(train_loss).all():
        print(f"  🔹 Final Training Loss: {train_loss[~np.isnan(train_loss)][-1]:.4f}")
    if not np.isnan(eval_loss).all():
        print(f"  🔹 Final Eval Loss: {eval_loss[~np.isnan(eval_loss)][-1]:.4f}")
    if not np.isnan(acc).all():
        print(f"  🔹 Final Eval Accuracy: {acc[~np.isnan(acc)][-1]:.4f}")
    if not np.isnan(f1).all():
        print(f"  🔹 Final Eval F1: {f1[~np.isnan(f1)][-1]:.4f}")
    if not np.isnan(lr).all():
        print(f"  🔹 Final Learning Rate: {lr[~np.isnan(lr)][-1]:.6f}")


plot_training_logs(f"outputs/models/bert_finetuned_openalex/checkpoint-500/",
                   save_path=f"outputs/plots/bert_training_metrics.png")
