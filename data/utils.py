import pandas as pd
from sklearn.model_selection import train_test_split


def get_n_rows_datasets(csv_path: str = "final/data.csv",
                        label_col: str = "major_topics",
                        text_col: str = "clean_abstract", rows=100):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col]).sample(n=rows).copy()
    return df


def get_n_rows_train_test_df(csv_path: str = "final/data.csv",
                            label_col: str = "major_topics",
                            text_col: str = "clean_abstract", rows=100):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[text_col, label_col]).sample(n=rows).copy()

        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df[label_col], random_state=42
        )
        return train_df, test_df, df
