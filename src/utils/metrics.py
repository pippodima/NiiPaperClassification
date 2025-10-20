from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_metrics(df, ground_col_label="major_topics", prediction_label="predicted_category"):
    accuracy = accuracy_score(df[ground_col_label], df[prediction_label])
    report = classification_report(df['major_topics'], df['predicted_category'])
    cm = confusion_matrix(df['major_topics'], df['predicted_category'], labels=df['major_topics'].unique())

    sns.heatmap(cm, annot=True, fmt="d", xticklabels=df['major_topics'].unique(),
                yticklabels=df['major_topics'].unique())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("outputs/plots/confusion_matrix.png")

    return accuracy, report
