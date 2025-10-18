import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple MPS GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚙️ Using CPU")
    return device


def get_model(model_name: str, num_labels: int):
    device = get_device()
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0)
    }
