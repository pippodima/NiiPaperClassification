import argparse
import ast
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP
from plot_clusters import *
import hdbscan
import ollama
import torch
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# DATA LOADING
# ============================================================================


def load_embeddings_parquet(path: str) -> tuple:
    print(f"📂 Loading {path}")
    df = pd.read_parquet(path)
    print(f"✅ Loaded {len(df)} rows")
    tqdm.pandas(desc="Parsing embeddings")
    embeddings = np.stack(
        df["embedding"].progress_apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))
    )
    df["combined_text"] = df["title"].fillna("") + ". " + df["clean_abstract"].fillna("")
    return df, normalize(embeddings)


# ============================================================================
# CLUSTERING
# ============================================================================

def perform_umap(embeddings, n_neighbors=15, min_dist=0.1, n_components=10, random_state=42):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔧 Running UMAP on {device}")
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
        verbose=False,
        low_memory=True
    )
    return umap_model.fit_transform(embeddings)


def perform_hdbscan(embeddings, min_cluster_size=30, min_samples=5, epsilon=0.3):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="leaf",
        gen_min_span_tree=False
    )
    return clusterer.fit_predict(embeddings)


def auto_optimize_clustering(embeddings):
    # 🏆 Best config: {'neighbors': 5, 'min_dist': 0.2, 'cluster_size': 10}(score=132.316)
    configs = []
    for n in [5]:  # [5, 10, 20]
        for d in [0.2]:  # [0.05, 0.1, 0.2]
            for c in [10]:  # [10, 20, 50]
                configs.append({"neighbors": n, "min_dist": d, "cluster_size": c})

    best_score, best_cfg, best_labels, best_umap = -1, None, None, None

    for cfg in tqdm(configs, desc="🔎 Auto-tuning clustering"):
        reduced = perform_umap(embeddings, n_neighbors=cfg["neighbors"], min_dist=cfg["min_dist"])
        labels = perform_hdbscan(reduced, min_cluster_size=cfg["cluster_size"])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        outliers = list(labels).count(-1)
        score = n_clusters / (1 + outliers / max(1, len(labels)))
        if score > best_score:
            best_score, best_cfg, best_labels, best_umap = score, cfg, labels, reduced

    print(f"🏆 Best config: {best_cfg} (score={best_score:.3f})")
    return best_umap, best_labels, best_cfg


# ============================================================================
# CLUSTER LABELING & SUMMARIES
# ============================================================================

def extract_top_tfidf_terms(texts, top_n=10):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return ["placeholdertext"]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        x = vectorizer.fit_transform(texts)
    except ValueError:
        return ["placeholdertext"]
    tfidf_sum = np.asarray(x.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_indices = np.argsort(tfidf_sum)[::-1][:top_n]
    return terms[top_indices].tolist()


def generate_llm_label(keywords, llm_model="mistral:7b"):
    prompt = f"""You are a research topic summarizer.
The following keywords describe a scientific topic cluster:
{keywords}
Return only a concise 2–4 word topic name."""
    try:
        response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
        name = response["message"]["content"].strip()
        name = name.replace("<think>", "").split("</think>")[-1].strip()
        return name
    except Exception:
        return None


def generate_cluster_summary(texts, llm_model="mistral:7b"):
    joined = "\n".join(texts[:5])
    prompt = f"""Summarize the following research abstracts in one concise sentence:
{joined}
Return only the summary."""
    try:
        response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
        name = response["message"]["content"].strip()
        name = name.replace("<think>", "").split("</think>")[-1].strip()
        return name
    except Exception:
        return None


def label_and_summarize_clusters(df, labels, text_col="combined_text",
                                 llm_model="mistral:7b", fast=False):
    df["cluster_id"] = labels
    cluster_info = []

    for cid in tqdm(sorted(set(labels)), desc="🧠 Labeling clusters"):
        if cid == -1:
            continue
        cluster_texts = df.loc[df["cluster_id"] == cid, text_col].dropna().tolist()
        if not cluster_texts:
            continue
        top_terms = extract_top_tfidf_terms(cluster_texts, 10)
        keywords = ", ".join(top_terms)
        name = " / ".join(top_terms[:3])
        summary = ""
        if not fast and len(cluster_texts) >= 500:
            llm_name = generate_llm_label(keywords, llm_model)
            name = llm_name or name
            summary = generate_cluster_summary(cluster_texts, llm_model) if not fast else None
        cluster_info.append({
            "cluster_id": cid,
            "keywords": top_terms,
            "name": name,
            "summary": summary or "",
            "size": len(cluster_texts)
        })

    info_df = pd.DataFrame(cluster_info)
    df = df.merge(info_df[["cluster_id", "name", "summary"]], on="cluster_id", how="left")
    return df, info_df


# ============================================================================
# RE-LABELING MODE
# ============================================================================

def relabel_clusters(input_path, llm_model="mistral:7b", fast=False):
    print(f"♻️ Re-labeling clusters in {input_path}")
    df = pd.read_parquet(input_path)
    if "cluster_id" not in df.columns:
        raise ValueError("❌ Missing cluster_id column; cannot relabel.")
    labels = df["cluster_id"].to_numpy()
    unlabeled = df[df["name"].isna() | (df["name"] == "")]
    if unlabeled.empty:
        print("✅ All clusters already labeled.")
        return
    df_updated, info_df = label_and_summarize_clusters(
        df, labels, fast=fast, llm_model=llm_model
    )
    out_path = input_path.replace(".parquet", "_relabeled.parquet")
    df_updated.to_parquet(out_path, index=False)
    print(f"💾 Saved updated file → {out_path}")
    info_df.to_parquet(out_path.replace(".parquet", "_metadata.parquet"), index=False)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings and label topics")
    parser.add_argument("--input", type=str, default="data/final/data_sci.parquet")
    parser.add_argument("--output", type=str, default="data/results/")
    parser.add_argument("--fast", action="store_true", help="Skip LLM labeling and summaries")
    parser.add_argument("--relabel", type=str, help="Path to clustered parquet file to relabel only")
    args = parser.parse_args()

    if args.relabel:
        relabel_clusters(args.relabel, fast=args.fast)
        return

    df, embeddings = load_embeddings_parquet(args.input)
    umap_embeddings, labels, best_cfg = auto_optimize_clustering(embeddings)

    df_labeled, cluster_info = label_and_summarize_clusters(
        df, labels, llm_model="qwen3:1.7b", fast=args.fast
    )

    # Save results
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "clustered_data.parquet")
    df_labeled.to_parquet(out_path, index=False)
    cluster_info.to_parquet(os.path.join(args.output, "cluster_metadata.parquet"), index=False)
    print(f"💾 Saved labeled data → {out_path}")

    # Plots
    plot_clusters_static(umap_embeddings, labels, os.path.join(args.output, "umap_clusters.png"))
    plot_clusters_interactive(df_labeled, umap_embeddings, os.path.join(args.output, "umap_clusters.html"))

    # Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "best_config": best_cfg,
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "n_outliers": int(list(labels).count(-1)),
        "fast_mode": args.fast,
    }
    with open(os.path.join(args.output, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("✅ Clustering complete.")


if __name__ == "__main__":
    main()
