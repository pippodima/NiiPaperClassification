from sklearn.cluster import KMeans
from plot_clusters import *


def create_cluster():
    df = pd.read_csv("../data/final/nii.csv")

    df["embedding"] = df["embedding"].apply(lambda x: np.array([float(i) for i in x.strip("[]").split()]))
    X = np.vstack(df["embedding"].values)

    # Run KMeans
    kmeans = KMeans(n_clusters=10, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)
    df.to_csv("tmp.csv", index=False)


def main():
    df = pd.read_csv("tmp.csv")
    cluster_counts = df["cluster"].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"Class {cluster} has {count} papers")


if __name__ == "__main__":
    create_cluster()
    # plot()
    # plot_3d()
    plot_interactive_3d()
    main()
