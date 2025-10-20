from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_metrics(df, ground_col_label="major_topics", prediction_label="predicted_category"):
    # Extract unique labels in a fixed order
    labels = sorted(df[ground_col_label].unique())

    # Compute metrics
    accuracy = accuracy_score(df[ground_col_label], df[prediction_label])
    report = classification_report(df[ground_col_label], df[prediction_label], zero_division=0)
    cm = confusion_matrix(df[ground_col_label], df[prediction_label], labels=labels)

    # Create larger figure to fit long labels
    plt.figure(figsize=(10, 8))

    # Plot confusion matrix
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )

    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)

    # Rotate tick labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Adjust layout so labels don’t get cut off
    plt.tight_layout()

    # Save the figure
    plt.savefig("outputs/plots/confusion_matrix.png", dpi=300)
    plt.close()

    return accuracy, report
