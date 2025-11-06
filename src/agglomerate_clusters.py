"""
aggregate_clusters_hierarchical.py
==================================
Iteratively merges clusters from previous experiments into higher-level,
more generic groups based on cosine similarity between cluster centroids.

- Starts at 0.8 similarity and decreases by 0.1 until 0.5.
- Merges all connected clusters (not just pairs) above the threshold.
- Optionally uses cluster metadata to enrich LLM naming.
- Uses Ollama (qwen3:1.7b) to generate semantic names.
- Saves per-threshold centroid files and a full merge history.
- Displays tqdm progress bars and prints a plain-text summary table.

Usage:
    python aggregate_clusters_hierarchical.py \
        --input-dir data/results \
        --naming-method llm \
        --save-results True
"""

import os
import json
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import ollama


# ============================================================================
# CONFIG FLAGS
# ============================================================================

USE_METADATA_FOR_NAMING = True  # Toggle: include metadata keywords in LLM naming
OUTPUT_DIR = "../data/aggregated_hierarchical"
MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 0.9
THRESHOLD_STEP = 0.1
LLM_MODEL = "qwen3:1.7b"


# ============================================================================
# LLM NAMING
# ============================================================================

def generate_llm_cluster_name(subcluster_names, keywords=None, model=LLM_MODEL):
    """Generate an umbrella cluster name using Ollama."""
    keywords_text = f"Keywords: {', '.join(keywords)}" if keywords else ""
    prompt = f"""
You are an expert research topic summarizer.
Below are names of related clusters:
{subcluster_names}

{keywords_text}

Suggest a concise 2–4 word topic name that represents them all.
Examples: 'Cancer Genomics', 'Climate Studies', 'Language Modeling'.
Only output the name.
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt.strip()}]
        )
        name = response["message"]["content"].strip()
        name = name.replace("<think>", "").split("</think>")[-1].strip()
        return name
    except Exception as e:
        print(f"⚠️ LLM naming failed: {e}")
        return "General"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_centroids(input_dir: str):
    """Load all centroid files and associated metadata."""
    centroid_files = sorted(f for f in os.listdir(input_dir) if f.startswith("centroids_") and f.endswith(".csv.gz"))
    meta_files = sorted(f for f in os.listdir(input_dir) if f.startswith("metadata_") and f.endswith(".json"))

    if not centroid_files:
        raise FileNotFoundError("❌ No centroid files found in input directory.")

    all_centroids = []
    all_meta = {}

    for cf, mf in zip(centroid_files, meta_files):
        centroids = pd.read_csv(os.path.join(input_dir, cf), compression="gzip")
        with open(os.path.join(input_dir, mf), "r", encoding="utf-8") as f:
            meta = json.load(f)

        config_name = f"{meta['config']['umap_neighbors']}_{meta['config']['hdb_min_cluster_size']}"
        centroids["source_config"] = config_name
        centroids["cluster_id"] = centroids.index
        centroids["cluster_name"] = centroids["cluster_id"].map(meta["cluster_names"])
        all_centroids.append(centroids)
        all_meta[config_name] = meta

    df = pd.concat(all_centroids, ignore_index=True)
    print(f"✅ Loaded {len(df)} clusters across {len(all_meta)} configs.")
    return df, all_meta


# ============================================================================
# CLUSTER MERGING LOGIC
# ============================================================================

def find_connected_components(sim_matrix, threshold):
    """Find connected components of clusters based on similarity threshold."""
    n = sim_matrix.shape[0]
    visited = set()
    components = []

    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        component = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            neighbors = np.where(sim_matrix[node] >= threshold)[0]
            for nb in neighbors:
                if nb not in visited:
                    stack.append(nb)
        components.append(list(component))
    return components


def merge_clusters(df, threshold, metadata=None):
    """Perform one merging iteration at given threshold."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = df[numeric_cols].values

    # Compute similarity matrix
    tqdm.write(f"🔍 Computing similarities (threshold={threshold}) ...")
    # Ensure features have no NaN or inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    sim = cosine_similarity(features)

    # Find connected components
    groups = find_connected_components(sim, threshold)
    merges = [g for g in groups if len(g) > 1]
    tqdm.write(f"🔗 Found {len(merges)} merge groups.")

    if not merges:
        return df, [], 0

    new_rows = []
    merge_records = []

    for mid, group in enumerate(tqdm(merges, desc=f"Merging groups ≥{threshold}")):
        sub_df = df.iloc[group]
        mean_vec = np.nan_to_num(sub_df[numeric_cols].mean(axis=0).values, nan=0.0, posinf=0.0, neginf=0.0)
        sub_names = sub_df["cluster_name"].dropna().astype(str).tolist()

        # Get keywords from metadata if enabled
        keywords = []
        if USE_METADATA_FOR_NAMING and metadata:
            for name in sub_df["source_config"].unique():
                meta = metadata.get(name, {})
                for _, val in meta.get("cluster_names", {}).items():
                    if isinstance(val, str):
                        keywords.extend(val.split())

        merged_name = generate_llm_cluster_name(", ".join(sub_names), keywords=keywords)
        new_row = dict(zip(numeric_cols, mean_vec))
        new_row.update({
            "cluster_name": merged_name,
            "merged_from": group,
            "threshold": threshold
        })
        new_rows.append(new_row)
        merge_records.append({
            "threshold": threshold,
            "merged_from": group,
            "new_name": merged_name
        })

    # Keep unmerged clusters
    merged_idxs = {i for g in merges for i in g}
    remaining_df = df.iloc[[i for i in range(len(df)) if i not in merged_idxs]]
    new_df = pd.concat([remaining_df, pd.DataFrame(new_rows)], ignore_index=True)

    return new_df, merge_records, len(merges)


# ============================================================================
# MAIN ITERATIVE AGGREGATION LOOP
# ============================================================================

def iterative_merge(df, metadata):
    """Iteratively merge clusters across decreasing thresholds."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = defaultdict(list)
    summary = []

    threshold = MAX_THRESHOLD
    iteration = 0

    while threshold >= MIN_THRESHOLD:
        tqdm.write(f"\n🚀 Threshold = {threshold:.1f}")
        merges_done = 0
        merge_records_all = []

        while True:
            df_new, merge_records, n_merges = merge_clusters(df, threshold, metadata)
            if n_merges == 0:
                break
            merges_done += n_merges
            merge_records_all.extend(merge_records)
            df = df_new

        # Save results once per threshold
        if merges_done > 0:
            level_path = os.path.join(OUTPUT_DIR, f"centroids_level_{threshold:.1f}.csv.gz")
            df.to_csv(level_path, index=False, compression="gzip")
            tqdm.write(f"💾 Saved merged centroids → {level_path}")
            history[f"level_{threshold:.1f}"] = merge_records_all
            summary.append({
                "threshold": threshold,
                "iteration": iteration,
                "n_merges": merges_done,
                "n_clusters_remaining": len(df)
            })

        threshold = round(threshold - THRESHOLD_STEP, 1)
        iteration += 1

    # Save final results
    final_path = os.path.join(OUTPUT_DIR, "final_hierarchical_centroids.csv.gz")
    df.to_csv(final_path, index=False, compression="gzip")

    def make_json_safe(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    meta_path = os.path.join(OUTPUT_DIR, "hierarchical_merge_history.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(history), f, indent=2, ensure_ascii=False)

    tqdm.write(f"\n✅ Final centroids saved → {final_path}")
    tqdm.write(f"🧾 Merge history saved → {meta_path}")

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Cluster Aggregation")
    parser.add_argument("--input-dir", type=str, default="../data/results", help="Directory with previous clustering results")
    parser.add_argument("--naming-method", type=str, choices=["tfidf", "llm"], default="llm", help="Cluster naming method")
    parser.add_argument("--save-results", type=bool, default=True, help="Save intermediate results")
    args = parser.parse_args()

    tqdm.write("📂 Loading data ...")
    df, metadata = load_centroids(args.input_dir)

    tqdm.write("🏗️ Starting hierarchical aggregation ...")
    summary = iterative_merge(df, metadata)

    # Print plain-text summary table
    print("\n================= SUMMARY =================")
    print(f"{'Threshold':<12} {'Merges':<10} {'Clusters Remaining':<20}")
    print("-------------------------------------------")
    for s in summary:
        print(f"{s['threshold']:<12.1f} {s['n_merges']:<10d} {s['n_clusters_remaining']:<20d}")
    print("===========================================")
    print("✅ Hierarchical aggregation complete!")


if __name__ == "__main__":
    main()
