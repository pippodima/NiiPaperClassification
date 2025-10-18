from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from datasets import Value
import numpy as np


def get_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_and_encode_labels(
    train_dataset,
    test_dataset,
    df,
    label_col: str = "major_topics",
    model_name="bert-base-uncased"
):
    tokenizer = get_tokenizer(model_name)
    le = LabelEncoder()
    le.fit(df[label_col])

    def tokenize(batch):
        return tokenizer(
            batch["abstract"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    # Tokenize text
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Map the correct labels from df to each dataset
    train_labels = le.transform(df.loc[train_dataset["__index_level_0__"], label_col])
    test_labels = le.transform(df.loc[test_dataset["__index_level_0__"], label_col])

    # Ensure labels are int64 (not float)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_labels = np.array(test_labels, dtype=np.int64)

    # Add labels back to the datasets
    train_dataset = train_dataset.add_column("labels", train_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    # Make sure Hugging Face knows these are int64
    train_dataset = train_dataset.cast_column("labels", Value("int64"))
    test_dataset = test_dataset.cast_column("labels", Value("int64"))

    return train_dataset, test_dataset, tokenizer, le
