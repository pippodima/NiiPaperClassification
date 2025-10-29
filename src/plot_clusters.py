from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns


def plot():
    df = pd.read_csv("tmp.csv")
    df["embedding"] = df["embedding"].apply(lambda x: np.array([float(i) for i in x.strip("[]").split()]))
    X = np.vstack(df["embedding"].values)

    X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)

    plt.scatter(X_2d[:,0], X_2d[:,1], c=df["cluster"], cmap="viridis")
    plt.title("Embedding Clusters (t-SNE)")
    plt.show()


def plot_3d():
    df = pd.read_csv("tmp.csv")
    df["embedding"] = df["embedding"].apply(lambda x: np.array([float(i) for i in x.strip("[]").split()]))
    X = np.vstack(df["embedding"].values)

    # Reduce to 3D
    X_3d = TSNE(n_components=3, random_state=42).fit_transform(X)

    # Create 3D plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c=df["cluster"], cmap="viridis", s=50)

    ax.set_xlabel("TSNE 1")
    ax.set_ylabel("TSNE 2")
    ax.set_zlabel("TSNE 3")
    plt.title("3D Embedding Clusters (t-SNE)")

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Cluster")

    plt.show()


def plot_interactive_3d(df, cluster_col="cluster", title_col="titles", perplexity=30, random_state=42):

    # Convert embeddings from space-separated string to numpy arrays
    df["embedding"] = df["embedding"].apply(lambda x: np.array([float(i) for i in x.strip("[]").split()]))
    X = np.vstack(df["embedding"].values)

    # Reduce to 3D using t-SNE
    X_3d = TSNE(n_components=3, perplexity=perplexity, random_state=random_state).fit_transform(X)

    # Add 3D coordinates to DataFrame
    df["x"], df["y"], df["z"] = X_3d[:, 0], X_3d[:, 1], X_3d[:, 2]

    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        df, x="x", y="y", z="z",
        color=cluster_col,
        hover_data=[title_col],
        color_continuous_scale="Viridis"
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(title="3D Interactive Embedding Clusters (t-SNE)")
    fig.show()


def plot_embedding(umap_embeddings, labels, title="Clusters of Papers by Abstract Similarity"):
    # Prepare
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    # Define a nice color palette (Spectral sometimes looks harsh)
    palette = sns.color_palette("husl", len(unique_labels))

    # Outliers (-1) get a light gray color
    colors = [palette[label % len(palette)] if label != -1 else (0.8, 0.8, 0.8) for label in labels]

    plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=colors,
        s=12,               # smaller points
        alpha=0.6,          # semi-transparent
        linewidths=0,
        edgecolors="none"
    )

    # Style
    plt.title(title, fontsize=14, weight="bold", pad=12)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # Add subtle white background and frame
    plt.gca().set_facecolor("#fafafa")
    plt.box(True)

    plt.show()


def plot_embedding_interactive(df, umap_embeddings, labels, i, title="Clusters of Papers by Abstract Similarity"):
    # Add embeddings and labels to DataFrame
    df = df.copy()
    df["x"] = umap_embeddings[:, 0]
    df["y"] = umap_embeddings[:, 1]
    df["cluster"] = labels

    # Prepare hover text
    df["hover_text"] = df["title"].fillna("No title").astype(str).str.slice(0, 150)

    # Outliers (-1) → gray color
    df["cluster_str"] = df["cluster"].astype(str)
    df.loc[df["cluster"] == -1, "cluster_str"] = "Outlier"

    # Create the interactive scatter plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_str",
        hover_data={"title": True, "cluster_str": True},
        hover_name="hover_text",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=title,
        width=1000,
        height=800
    )

    # Style
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        title_font=dict(size=20, family="Arial", color="black"),
        legend_title_text="Cluster ID",
    )

    fig.show()
    fig.write_html(f"outputs/html/clusters_config{i}.html")
