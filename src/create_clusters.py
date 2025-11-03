import ast
import re
import unicodedata
import mojimoji
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
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

def load_embeddings(path, subset="full"):
    """Load gzipped CSV with stringified embeddings → numpy array.
    subset: 'full', 'scientific_paper', or 'diagnostic_report'
    """
    print(f"📂 Loading dataset from {path} ...")
    df = pd.read_csv(path, compression="gzip")
    print(f"✅ Loaded {len(df)} rows")

    # Subset based on 'type' column if needed
    if subset != "full":
        if "type" not in df.columns:
            raise ValueError("❌ Dataset must have a 'type' column for subsetting.")
        df = df[df["type"] == subset]
        print(f"🔍 Subset to {subset}: {len(df)} rows")

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

def perform_umap(embeddings, n_neighbors=100, min_dist=0.1, n_components=10, random_state=42):
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
        x = vectorizer.fit_transform(cluster_texts)
        tfidf_sum = np.asarray(x.sum(axis=0)).ravel()
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
            topic_name = topic_name.replace("<think>", "").split("</think>")[-1].strip()
            cluster_names[cluster_id] = topic_name or " / ".join(top_terms[:3])
        except Exception as e:
            print(f"⚠️ Ollama naming failed for cluster {cluster_id}: {e}")
            cluster_names[cluster_id] = " / ".join(top_terms[:3])

    df["cluster_name"] = df["cluster_id"].map(cluster_names)
    return df, cluster_names


# ============================================================
# 🔁 Experiment Runner
# ============================================================

def try_multiple_configurations(df, embeddings, configs, name_method="llm", saveFig=False):
    """
    Run clustering under multiple configurations and compare results.
    """
    results = []

    for i, cfg in enumerate(configs, start=1):
        print(f"\n{'='*60}")
        print(f"⚙️  Running configuration {i}/{len(configs)}: {cfg}")
        print(f"{'='*60}")

        reduced_embeddings = perform_umap(
            embeddings,
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

        umap_2d = perform_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=2)
        df_temp = df.copy()
        df_temp["cluster_id"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        print(f"📊 Clusters: {n_clusters} | 🚫 Outliers: {n_outliers}")

        df_temp = clean_for_tfidf(df, text_col="combined_text")

        # Label clusters
        df_labeled, cluster_names = name_clusters(
            df_temp,
            labels,
            text_col="cleaned_for_tfidf",
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
            plot_embedding(umap_embeddings=umap_2d,
                           labels=labels,
                           neighbors=cfg.get("umap_neighbors"),
                           cluster_size=cfg.get("hdb_min_cluster_size"),
                           save=saveFig)
            plot_embedding_interactive(df=df_temp,
                                       umap_embeddings=umap_2d,
                                       labels=labels,
                                       neighbors=cfg.get("umap_neighbors"),
                                       cluster_size=cfg.get("hdb_min_cluster_size"),
                                       save=saveFig)
        except Exception as e:
            print(f"⚠️ Skipped plotting: {e}")

        results.append({
            "config": cfg,
            "df": df_labeled,
            "labels": labels,
            "cluster_names": cluster_names
        })

    return results


def clean_for_tfidf(df, text_col="combined_text"):
    # Common boilerplate or metadata terms to drop
    BAD_WORDS = {
        "pdf", "article", "text", "type", "source", "application",
        "identifier", "pp", "doc", "document", "資料番号", "論文",
        "著者", "journal", "abstract", "file", "doi", "url"
    }

    def _clean_one(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1️⃣ Remove HTML/XML tags
        text = BeautifulSoup(text, "lxml").get_text(separator=" ")

        # 2️⃣ Normalize Unicode (NFKC) and convert fullwidth to halfwidth for consistency
        text = unicodedata.normalize("NFKC", text)
        text = mojimoji.zen_to_han(text, kana=False)

        # 3️⃣ Remove URLs, emails, file refs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"\b\w+\.(pdf|docx?|xlsx?|txt)\b", " ", text)

        # 4️⃣ Lowercase English text
        text = text.lower()

        # 5️⃣ Keep only useful characters (Japanese, English, digits)
        text = re.sub(r"[^ぁ-んァ-ン一-龥a-z0-9\s]", " ", text)

        # 6️⃣ Remove boilerplate terms
        pattern = r"\b(" + "|".join(map(re.escape, BAD_WORDS)) + r")\b"
        text = re.sub(pattern, " ", text)

        # 7️⃣ Remove isolated one-letter tokens
        text = re.sub(r"\b[a-z]\b", " ", text)

        # 8️⃣ Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    df["cleaned_for_tfidf"] = df[text_col].astype(str).map(_clean_one)
    return df


# ============================================================
# 🚀 Main
# ============================================================

def main():
    save = False
    saveFig = False
    input_path = "data/final/data.csv.gz"
    df, embeddings = load_embeddings(path=input_path, subset="scientific_paper")

    # Define configs to try

    configs = [
        {"umap_neighbors": 10, "hdb_min_cluster_size": 50}
        # {"umap_neighbors": 30, "hdb_min_cluster_size": 400},
        # {"umap_neighbors": 50, "hdb_min_cluster_size": 500},
        # {"umap_neighbors": 75, "hdb_min_cluster_size": 1000},
        # {"umap_neighbors": 100, "hdb_min_cluster_size": 1250},
        # {"umap_neighbors": 125, "hdb_min_cluster_size": 1500}
    ]
    # Choose naming method: "tfidf" (fast) or "llm" (Ollama semantic)
    results = try_multiple_configurations(df, embeddings, configs, name_method="tfidf", saveFig=saveFig)

    if save:
        for i, res in enumerate(results, start=1):
            name = f"data/final/clusters_config_{res['config']['umap_neighbors']}neighbors_{res['config']['hdb_min_cluster_size']}cluster_size.csv.gz"
            print(f"💾 Saving {name}")
            res["df"].to_csv(name, index=False, compression="gzip")

        print("\n✅ All experiments complete!")


if __name__ == "__main__":
    main()
