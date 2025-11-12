import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import hvplot.pandas
from bokeh import palettes
import colorcet as cc
import datashader as ds
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread
from bokeh.resources import CDN
from bokeh.embed import file_html


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


def plot_clusters_interactive(df, embeddings, output_path, sample_hover=2000):
    import pandas as pd
    import numpy as np
    import hvplot.pandas  # noqa
    import holoviews as hv
    import datashader as ds
    import colorcet as cc
    from holoviews.operation.datashader import datashade, dynspread
    from bokeh.resources import CDN
    from bokeh.embed import file_html
    from bokeh import palettes

    hv.extension("bokeh")

    df_plot = df.copy()
    df_plot["x"] = embeddings[:, 0]
    df_plot["y"] = embeddings[:, 1]
    df_plot["cluster_id"] = df_plot["cluster_id"].astype(str)

    n_clusters = df_plot["cluster_id"].nunique()

    # Palette logic
    if n_clusters <= 20:
        palette = palettes.Category20[n_clusters]
    elif n_clusters <= 256:
        palette = cc.glasbey[:n_clusters]
    else:
        from matplotlib.cm import get_cmap
        cmap = get_cmap("tab20", n_clusters)
        palette = [
            f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            for r, g, b, _ in cmap(np.linspace(0, 1, n_clusters))
        ]

    cats = sorted(df_plot["cluster_id"].unique())
    color_key = dict(zip(cats, palette))

    # Datashader background (all points)
    points = hv.Points(df_plot, kdims=["x", "y"], vdims=["cluster_id"])
    aggregator = ds.count_cat("cluster_id")

    shaded = datashade(
        points,
        aggregator=aggregator,
        color_key=color_key,
        width=1000,
        height=800,
        min_alpha=80,
    )
    shaded = dynspread(shaded, threshold=0.7, max_px=3)

    # ✨ Hover layer (sampled subset)
    if len(df_plot) > sample_hover:
        df_hover = df_plot.sample(n=sample_hover, random_state=42)
    else:
        df_hover = df_plot

    hover_points = hv.Points(
        df_hover, kdims=["x", "y"],
        vdims=["cluster_id", "title"]
    ).opts(
        color="cluster_id",
        size=6,
        alpha=0.8,
        cmap=palette,
        tools=["hover"],
        hover_fill_color="white",
        hover_line_color="black",
        legend_position="right",
    )

    layout = (shaded * hover_points).opts(
        title=f"UMAP Clusters ({len(df_plot):,} points, {n_clusters} clusters)",
        xlabel="UMAP-1",
        ylabel="UMAP-2",
        frame_width=900,
        frame_height=700,
        show_grid=False,
    )

    html = file_html(hv.render(layout), CDN, "UMAP Clusters")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Saved interactive Datashader plot with hover → {output_path}")
