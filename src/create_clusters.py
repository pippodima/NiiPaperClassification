import ast

from umap import UMAP
import hdbscan
import pandas as pd
from plot_clusters import plot_embedding
import numpy as np
from sklearn.preprocessing import normalize


def main():
    df = pd.read_csv("data/final/data_with_embeddings.csv")

    print(len(df["embedding"]))

    # embeddings = np.stack(df["embedding"].apply(lambda x: np.array([float(i) for i in x.strip("[]").split()])))
    embeddings = np.stack(df["embedding"].apply(lambda x: np.array(ast.literal_eval(x))))

    embeddings_norm = normalize(embeddings)

    umap_for_cluster = UMAP(
        n_neighbors=35,  # larger → capture global structure
        min_dist=0.1,  # moderate
        n_components=15,
        random_state=42
    )
    reduced_embeddings = umap_for_cluster.fit_transform(embeddings_norm)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=70,  # larger clusters → fewer major topics
        min_samples=8,  # strict density → cohesive clusters
        cluster_selection_epsilon=0.15  # merge nearby clusters
    )
    labels = clusterer.fit_predict(reduced_embeddings)

    umap_2d = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    umap_embeddings = umap_2d.fit_transform(embeddings_norm)

    plot_embedding(umap_embeddings, labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    print(f"\nNumber of clusters (excluding outliers): {n_clusters}")
    print(f"Number of outliers: {n_outliers}")

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        print(f"\nCluster {cluster_id}:")
        idx = np.where(labels == cluster_id)[0]
        for i in idx[:3]:  # show up to 3 samples per cluster
            title = df.iloc[i].get("title", "No title")
            print(f" - {title}")

    df["cluster_id"] = labels
    df.to_csv("data/final/data_clustered.csv", index=False)
    print("\nSaved clustered dataset → data/final/nii_clustered.csv")

    print(pd.Series(labels).value_counts().sort_index())


if __name__ == "__main__":
    main()
