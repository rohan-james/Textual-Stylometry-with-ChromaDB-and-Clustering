import chromadb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import seaborn as sns
import os

CHROMA_DB_PATH = "../chroma_db_data"
COLLECTION_NAME = "literary_styles"


def main():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"error retrieving collection: {e}")
        return

    results = collection.get(
        ids=collection.get()["ids"], include=["embeddings", "metadatas", "documents"]
    )

    if len(results["embeddings"]) == 0:
        print("no embeddings found in collection")
        return

    embeddings = np.array(results["embeddings"])
    metadatas = results["metadatas"]
    documents = results["documents"]

    df = pd.DataFrame(metadatas)
    df["embedding"] = list(embeddings)
    df["document"] = documents

    print(f"embedding shape: {embeddings.shape}")

    """
    Perform K-means clustering
    """
    k = 4

    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["kmeans_cluster"] = kmeans.fit_predict(embeddings)

        if k > 1 and len(embeddings) >= k:
            silhoutte_avg = silhouette_score(embeddings, df["kmeans_cluster"])
            print(f"K-Means Silhouette Score: {silhoutte_avg:.3f}")
        else:
            print("not enough clusters or data points to calculate the score")

        # print(df.groupby(['kmeans_cluster','author']).size().unstack(fill_value=0))

        print("cluster distribution by author: \n")
        for i in range(k):
            cluster_df = df[df["kmeans_cluster"] == i]
            top_authors = cluster_df["author"].value_counts().head(5)
            print(f"Cluster {i}")
            if not top_authors.empty:
                print(top_authors)
            else:
                print("empty cluster")
    except Exception as e:
        print(f"error during clustering: {e}")

    """
    Perform HDBSCAN clustering
    """
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=15, min_samples=5, prediction_data=True
        )
        df["hdbscan_cluster"] = clusterer.fit_predict(embeddings)
        hdbscan_clusters = df["hdbscan_cluster"].nunique() - (
            1 if -1 in df["hdbscan_cluster"].unique() else 0
        )

        print(f"found {hdbscan_clusters} HDBSCAN clusters")
        print(
            f"noise points: {np.sum(df['hdbscan_cluster'] == -1)} out of {len(df)} total"
        )

        clustered_df = df[df["hdbscan_cluster"] != -1]
        if not clustered_df.empty:
            print(
                clustered_df.groupby(["hdbscan_cluster", "author"])
                .size()
                .unstack(fill_value=0)
            )
        else:
            print("no non-noise clusters found")

        for i in sorted(clustered_df["hdbscan_cluster"].unique()):
            if i != -1:
                cluster_df = df[df["hdbscan_cluster"] == i]
                top_authors = cluster_df["author"].value_counts(0).head(5)
                if not top_authors.empty:
                    print(top_authors)
                else:
                    print("cluster {i} is empty")

    except Exception as e:
        print(f"error during hdbscan clustering: {e}")


if __name__ == "__main__":
    main()
