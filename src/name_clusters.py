import ast

import numpy as np
import pandas as pd
import ollama
from typing import Dict, List, Union


# ============================================================
#                CLUSTER NAMING PIPELINE
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_cluster_centroids(df: pd.DataFrame) -> pd.Series:
    """Compute the centroid (mean embedding) for each cluster."""
    return df.groupby("cluster_id")["embedding"].apply(
        lambda x: np.mean(np.vstack(x.to_list()), axis=0)
    )


def get_representative_docs(
    df: pd.DataFrame,
    centroids: pd.Series,
    top_k: int = 3
) -> pd.DataFrame:
    """
    Select top-k documents closest to the cluster centroid.

    Args:
        df: DataFrame with columns ['embedding', 'cluster_id']
        centroids: Series mapping cluster_id -> centroid embedding
        top_k: number of representative documents to select

    Returns:
        A filtered DataFrame with top_k docs per cluster.
    """
    df = df.copy()
    df["centroid_sim"] = df.apply(
        lambda r: cosine_similarity(r.embedding, centroids[r.cluster_id]), axis=1
    )
    top_docs = (
        df.sort_values("centroid_sim", ascending=False)
        .groupby("cluster_id")
        .head(top_k)
    )
    return top_docs


def merge_cluster_texts(
    top_docs: pd.DataFrame,
    text_columns: List[str] = ["title", "abstract"]
) -> pd.Series:
    """
    Merge representative texts (title + abstract) per cluster into a single string.

    Args:
        top_docs: representative papers (from get_representative_docs)
        text_columns: which columns to combine

    Returns:
        Series mapping cluster_id -> merged text
    """
    def combine_row(r):
        return ". ".join(str(r[c]) for c in text_columns if pd.notna(r[c]))

    return (
        top_docs.assign(text=top_docs.apply(combine_row, axis=1))
        .groupby("cluster_id")["text"]
        .apply(lambda g: "\n\n".join(g))
    )


def generate_cluster_label(
    cluster_id: Union[int, str],
    text: str,
    model: str = "qwen3:1.7b",
    temperature: float = 0.2
) -> str:
    """
    Use a local Ollama model to generate a concise topic label for a cluster.

    Args:
        cluster_id: cluster identifier
        text: combined text from representative docs
        model: name of the Ollama model to use
        temperature: creativity parameter (0–1)

    Returns:
        A concise topic label (string)
    """
    prompt = f"""
    You are naming scientific research topics.
    Below are titles and abstracts of 3 representative papers from the same cluster.

    Your task:
    - Identify the main shared theme of these papers.
    - Write a **concise topic label** (max 6 words).
    - Avoid generic terms like "study", "analysis", or "approach".
    - Return only the label, no explanation.

    TEXT:
    {text}
    """

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def generate_labels_for_all_clusters(
    df: pd.DataFrame,
    model: str = "qwen3:1.7b",
    top_k: int = 3,
    save_path: str = None
) -> Dict[int, str]:
    """
    Full pipeline: compute centroids, pick top docs, query Ollama, and return cluster labels.

    Args:
        df: DataFrame with ['title', 'abstract', 'embedding', 'cluster_id']
        model: Ollama model name
        top_k: number of representative docs per cluster
        save_path: optional path to save results (CSV)

    Returns:
        Dictionary mapping cluster_id -> label
    """
    centroids = compute_cluster_centroids(df)
    top_docs = get_representative_docs(df, centroids, top_k=top_k)
    cluster_texts = merge_cluster_texts(top_docs)

    labels = {}
    for cid, txt in cluster_texts.items():
        print(f"\n🧩 Generating label for cluster {cid}...")
        labels[cid] = generate_cluster_label(cid, txt, model=model)
        print(f"✅ Cluster {cid}: {labels[cid]}")

    if save_path:
        label_df = pd.DataFrame(list(labels.items()), columns=["cluster_id", "cluster_label"])
        label_df.to_csv(save_path, index=False)
        print(f"\n💾 Saved cluster labels to: {save_path}")

    return labels


def load_df(path="processed/rdf_results_final.csv.gz"):
    print("Loading df")
    df = pd.read_csv(path, compression="gzip")

    # Convert embedding strings like "[0.1, 0.2, 0.3]" → np.array([...])
    if isinstance(df.loc[0, "embedding"], str):
        df["embedding"] = df["embedding"].apply(
            lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
        )

    return df


if __name__ == "__main__":
    # Example: df = pd.read_pickle("papers_with_embeddings.pkl")
    # df must have: ['title', 'abstract', 'embedding', 'cluster_id']
    df = load_df("../data/final/clusters_config_125neighbors_2000cluster_size.csv.gz")
    # Run the labeling pipeline
    labels = generate_labels_for_all_clusters(
        df,
        model="qwen3:1.7b",
        top_k=3,
        save_path="../data/final/cluster_labels.csv.gz"
    )

    # Attach labels to df
    label_df = pd.DataFrame(list(labels.items()), columns=["cluster_id", "cluster_label"])
    df = df.merge(label_df, on="cluster_id", how="left")

    # Save enriched data
    df.to_csv("papers_with_labels.csv.gz", compression="gzip")
