import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import os


def plot_clusters_static(umap_embeddings, labels, output_path="plots/umap_clusters.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    plt.title("UMAP Clusters", fontsize=14, weight="bold", pad=12)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # Add subtle white background and frame
    plt.gca().set_facecolor("#fafafa")
    plt.box(True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    plt.show()
    print(f"📊 Saved static cluster plot → {output_path}")


def plot_clusters_interactive(df, umap_embeddings, output_path="plots/umap_clusters.html"):
    """Generate an interactive HTML plot with detailed hover info and cluster filtering."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_plot = df.copy()
    df_plot["x"] = umap_embeddings[:, 0]
    df_plot["y"] = umap_embeddings[:, 1]

    # Create custom hover text
    df_plot["hover_text"] = (
        "<b>Title:</b> " + df_plot["title"].astype(str) +
        "<br><b>Cluster:</b> " + df_plot["name"].fillna("Unknown").astype(str) +
        "<br><b>Summary:</b> " + df_plot["summary"].fillna("").astype(str)
    )

    # Build interactive scatter plot
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="name",
        hover_name="title",              # large title at top of hover box
        hover_data={
            "cluster_id": True,
            "summary": True,
            "x": False,
            "y": False
        },
        text=None,
        title="Interactive Cluster Visualization (Hover for Paper Details)",
        opacity=0.85,
        template="plotly_white",
    )

    # Replace default hover with our rich custom text
    fig.update_traces(marker=dict(size=6), hovertemplate=df_plot["hover_text"])

    # Add cluster filter dropdown
    cluster_names = sorted(df_plot["name"].dropna().unique())
    buttons = [
        dict(label="All Clusters",
             method="update",
             args=[{"visible": [True] * len(fig.data)},
                   {"title": "All Clusters"}])
    ]
    for cname in cluster_names:
        visible = [trace.name == cname for trace in fig.data]
        buttons.append(
            dict(label=cname,
                 method="update",
                 args=[{"visible": visible},
                       {"title": f"Cluster: {cname}"}])
        )
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.05,
            xanchor="left",
            y=1,
            yanchor="top"
        )],
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"🌐 Saved interactive HTML plot → {output_path}")
