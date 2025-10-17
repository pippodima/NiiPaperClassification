from transformers import pipeline
import pandas as pd
from tqdm import tqdm


# Load classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels
labels = [
    "Natural Sciences",
    "Engineering and Technology",
    "Medical and Health Sciences",
    "Agricultural Sciences",
    "Social Sciences",
    "Humanities",
]

# Load your dataset
df_original = pd.read_csv("../../data/final/data.csv")

# Subset of documents
df = df_original.sample(n=100, random_state=42).copy()

# Predict
predictions = []

for abstract in tqdm(df["abstract"], desc="Classifying abstracts"):
    result = classifier(abstract, labels)
    predictions.append(result["labels"][0])

df["llm_major_topic"] = predictions

# Save results
df.to_csv("../../data/predicted/openalex_llm_bart_large_mpli.csv", index=False)
