import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns


def plot_embedding(umap_embeddings, labels, neighbors, cluster_size, save=False, title="Clusters of Papers by Abstract Similarity"):
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

    if save:
        plt.savefig(f"outputs/plots/clusters_scientific_config_{neighbors}neighbors_{cluster_size}cluster_size.png")

    plt.show()


def plot_embedding_interactive(df, umap_embeddings, labels, neighbors, cluster_size, save=False, title="Clusters of Papers by Abstract Similarity"):
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
    if save:
        fig.write_html(f"outputs/html/clusters_config_scientific_{neighbors}neighbors_{cluster_size}cluster_size.html")
