import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load your CSV
df = pd.read_csv("../../data/predicted/openalex_llm_bart_large_mpli.csv")

# Labels
labels = [
    "Natural Sciences",
    "Engineering and Technology",
    "Medical and Health Sciences",
    "Agricultural Sciences",
    "Social Sciences",
    "Humanities",
]

y_true = df["major_topics"]
y_pred = df["llm_major_topic"]

# --- 1️⃣ Confusion Matrix Heatmap ---
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("LLM Classification Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 2️⃣ F1-Score per Class Bar Plot ---
report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
f1_scores = [report_dict[label]["f1-score"] for label in labels]

plt.figure(figsize=(10, 5))
sns.barplot(x=labels, y=f1_scores, palette="viridis")
plt.title("LLM Classification F1-Score per Major Topic")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
