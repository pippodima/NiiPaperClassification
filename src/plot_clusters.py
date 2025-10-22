from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


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


def plot_interactive_3d(csv_path="tmp.csv", cluster_col="cluster", title_col="titles", perplexity=30, random_state=42):
    # Load CSV
    df = pd.read_csv(csv_path)

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

