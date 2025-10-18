from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd


def prepare_datasets(csv_path: str = "../../data/final/data.csv", label_col: str = "major_topics", text_col: str = "abstract"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])

    # Encode labels as integers
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset, df, le


def prepare_small_datasets(csv_path: str = "../../data/final/data.csv", label_col: str = "major_topics", text_col: str = "abstract"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col]).sample(n=1000, random_state=42).copy()

    # Encode labels as integers
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df[label_col])

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=42
    )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, test_dataset, df
