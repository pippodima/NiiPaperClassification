import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import hdbscan
from tqdm import tqdm
from plot_clusters import plot_embedding, plot_embedding_interactive
import warnings
import ollama

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 🧩 Data Loading
# ============================================================

def load_embeddings(path):
    """Load gzipped CSV with stringified embeddings → numpy array."""
    print(f"📂 Loading dataset from {path} ...")
    df = pd.read_csv(path, compression="gzip")
    print(f"✅ Loaded {len(df)} rows")

    tqdm.pandas(desc="Parsing embeddings")
    embeddings = np.stack(df["embedding"].progress_apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32)))

    # Combine title + abstract for better topic naming
    if "title" in df.columns and "abstract" in df.columns:
        df["combined_text"] = df["title"].fillna("") + ". " + df["abstract"].fillna("")
    elif "title" in df.columns:
        df["combined_text"] = df["title"]
    else:
        raise ValueError("❌ Dataset must contain at least a 'title' column.")

    return df, embeddings


# ============================================================
# 🧬 Dimensionality Reduction + Clustering
# ============================================================

def perform_umap(embeddings, n_neighbors=100, min_dist=0.4, n_components=10, random_state=42):
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    return umap_model.fit_transform(embeddings)


def perform_hdbscan(embeddings, min_cluster_size=300, min_samples=10, epsilon=0.3):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method='leaf'
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


# ============================================================
# 🏷️ Cluster Labeling (TF-IDF + Ollama)
# ============================================================

def name_clusters(df, labels, text_col="combined_text", method="llm", top_n_words=10, llm_model="qwen3:1.7b"):
    """
    Assign meaningful names to clusters using TF-IDF or a local Ollama LLM.
    - method="tfidf": fast keyword-based naming
    - method="llm": semantic naming using Ollama model
    """
    df["cluster_id"] = labels
    cluster_names = {}
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    for cluster_id in tqdm(sorted(set(labels)), desc="🧠 Naming clusters"):
        if cluster_id == -1:
            cluster_names[cluster_id] = "Outliers"
            continue

        cluster_texts = df.loc[df["cluster_id"] == cluster_id, text_col].dropna().tolist()
        if not cluster_texts:
            cluster_names[cluster_id] = "Unknown"
            continue

        # Extract representative keywords via TF-IDF
        X = vectorizer.fit_transform(cluster_texts)
        tfidf_sum = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        top_terms = terms[np.argsort(tfidf_sum)[::-1][:top_n_words]]
        keywords = ", ".join(top_terms)

        if method == "tfidf":
            cluster_names[cluster_id] = " / ".join(top_terms[:3])
            continue

        # ---- Local Ollama LLM Mode ----
        prompt = f"""
        You are a research topic summarizer.
        The following keywords are representative of a cluster of academic papers:
        {keywords}

        Suggest a concise, human-readable topic name (2–4 words),
        e.g. "Cancer Biology", "Antarctic Climate Studies", "Japanese Linguistics".
        Only return the topic name.
        """

        try:
            response = ollama.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            topic_name = response["message"]["content"].strip()
            cluster_names[cluster_id] = topic_name or " / ".join(top_terms[:3])
        except Exception as e:
            print(f"⚠️ Ollama naming failed for cluster {cluster_id}: {e}")
            cluster_names[cluster_id] = " / ".join(top_terms[:3])

    df["cluster_name"] = df["cluster_id"].map(cluster_names)
    return df, cluster_names


# ============================================================
# 🔁 Experiment Runner
# ============================================================

def try_multiple_configurations(df, embeddings, configs, name_method="llm"):
    """
    Run clustering under multiple configurations and compare results.
    """
    results = []

    for i, cfg in enumerate(configs, start=1):
        print(f"\n{'='*60}")
        print(f"⚙️  Running configuration {i}/{len(configs)}: {cfg}")
        print(f"{'='*60}")

        embeddings_norm = normalize(embeddings)

        reduced_embeddings = perform_umap(
            embeddings_norm,
            n_neighbors=cfg.get("umap_neighbors", 100),
            min_dist=cfg.get("umap_min_dist", 0.4),
            n_components=cfg.get("umap_components", 10)
        )

        labels = perform_hdbscan(
            reduced_embeddings,
            min_cluster_size=cfg.get("hdb_min_cluster_size", 300),
            min_samples=cfg.get("hdb_min_samples", 10),
            epsilon=cfg.get("hdb_epsilon", 0.3)
        )

        umap_2d = perform_umap(embeddings_norm, n_neighbors=15, min_dist=0.1, n_components=2)
        df_temp = df.copy()
        df_temp["cluster_id"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        print(f"📊 Clusters: {n_clusters} | 🚫 Outliers: {n_outliers}")

        # Label clusters
        df_labeled, cluster_names = name_clusters(
            df_temp,
            labels,
            text_col="combined_text",
            method=name_method,
            llm_model="qwen3:1.7b"
        )

        print("🏷️ Cluster names:")
        for cid, name in cluster_names.items():
            if cid == -1:
                continue
            print(f"  - {cid}: {name}")

        # Plot clusters
        try:
            plot_embedding(umap_2d, labels)
            plot_embedding_interactive(df_temp, umap_2d, labels)
        except Exception as e:
            print(f"⚠️ Skipped plotting: {e}")

        results.append({
            "config": cfg,
            "df": df_labeled,
            "labels": labels,
            "cluster_names": cluster_names
        })

    return results


# ============================================================
# 🚀 Main
# ============================================================

def main():
    save = False
    input_path = "data/final/data_eng.csv.gz"
    df, embeddings = load_embeddings(input_path)

    # Define configs to try
    configs = [
        {"umap_neighbors": 50, "hdb_min_cluster_size": 200},
        {"umap_neighbors": 100, "hdb_min_cluster_size": 300},
        {"umap_neighbors": 150, "hdb_min_cluster_size": 400},
    ]

    # Choose naming method: "tfidf" (fast) or "llm" (Ollama semantic)
    results = try_multiple_configurations(df, embeddings, configs, name_method="llm")

    if save:
        for i, res in enumerate(results, start=1):
            name = f"data/final/data_clustered_config{i}.csv.gz"
            print(f"💾 Saving {name}")
            res["df"].to_csv(name, index=False, compression="gzip")

        print("\n✅ All experiments complete!")


if __name__ == "__main__":
    main()
