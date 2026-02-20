import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as sch
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & EDA
# ==========================================
print("--- Loading and Visualizing Data ---")
df = pd.read_csv('29-Country-data.csv')


def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=np.number).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)
    plt.figure(figsize=(n_cols * 3, n_rows * 2))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(title_prefix + col)
        plt.xlabel("");
        plt.ylabel("")
    plt.tight_layout()
    plt.show()


plot_all_histograms(df)

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# 2. DATA PREPROCESSING (SCALING & PCA)
# ==========================================
df_features = df.drop("country", axis=1)
scaler = MinMaxScaler()
df_raw = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

pca_temp = PCA()
pca_temp.fit(df_raw)
plt.figure(figsize=(8, 4))
plt.step(list(range(1, 10)), np.cumsum(pca_temp.explained_variance_ratio_))
plt.plot(np.cumsum(pca_temp.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components (PCA)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

pca = PCA(n_components=3)
df_pca = pd.DataFrame(pca.fit_transform(df_raw))

# ==========================================
# 3. HELPER FUNCTIONS & RESULTS LIST
# ==========================================
final_results = []


def plot_pca_clusters(data, labels, title):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=labels, palette='viridis', s=100, alpha=0.8,
                    legend='full')
    plt.title(title, fontsize=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_world_map(df_original, labels, title):
    map_df = pd.DataFrame({
        'Country': df_original['country'],
        'Cluster': [str(label) if label != -1 else 'Noise' for label in labels]
    }).sort_values(by="Cluster")

    fig = px.choropleth(map_df, locationmode="country names", locations="Country",
                        title=title, color="Cluster", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_geos(fitbounds="locations", visible=True)
    fig.show()


# ==========================================
# 4. K-MEANS CLUSTERING (PCA vs RAW)
# ==========================================
print("\n--- 1. K-MEANS EXPERIMENTS ---")

wcss = []
for k in range(1, 11):
    kmeans_temp = KMeans(n_clusters=k, init="k-means++", random_state=15).fit(df_pca)
    wcss.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xticks(range(1, 11))
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal K (PCA Data)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Model with PCA
kmeans_pca = KMeans(n_clusters=3, init="k-means++", random_state=15).fit(df_pca)
score_kmeans_pca = silhouette_score(df_pca, kmeans_pca.labels_)
print(f"-> K-Means (With PCA) Silhouette Score: {score_kmeans_pca:.4f}")
final_results.append({'Algorithm': 'K-Means', 'Data Type': 'With PCA (3D)', 'Silhouette Score': score_kmeans_pca})
plot_pca_clusters(df_pca, kmeans_pca.labels_, "K-Means Clustering With PCA (K=3)")
plot_world_map(df, kmeans_pca.labels_, "World Map: K-Means With PCA (K=3)")

# Model without PCA
kmeans_raw = KMeans(n_clusters=3, init='k-means++', n_init=50, max_iter=500, random_state=22).fit(df_raw)
score_kmeans_raw = silhouette_score(df_raw, kmeans_raw.labels_)
print(f"-> K-Means (Without PCA) Silhouette Score: {score_kmeans_raw:.4f}")
final_results.append({'Algorithm': 'K-Means', 'Data Type': 'Without PCA (9D)', 'Silhouette Score': score_kmeans_raw})
plot_pca_clusters(df_pca, kmeans_raw.labels_, "K-Means Clustering Without PCA (K=3)")
plot_world_map(df, kmeans_raw.labels_, "World Map: K-Means Without PCA (K=3)")

# ==========================================
# 5. DBSCAN CLUSTERING (PCA vs RAW)
# ==========================================
print("\n--- 2. DBSCAN EXPERIMENTS ---")

neighbors = NearestNeighbors(n_neighbors=2).fit(df_pca)
distances, _ = neighbors.kneighbors(df_pca)
distances = np.sort(distances[:, 1], axis=0)
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.title("K-Distance Graph (For EPS Selection - PCA Data)")
plt.xlabel("Data Points sorted by distance")
plt.ylabel("Epsilon (eps)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

eps_grid_pca = [0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
eps_grid_raw = [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.30, 0.35, 0.5, 0.75, 1.0, 1.5]
min_samples_grid = [4, 5, 6, 7, 8]

# Model with PCA
results_dbscan_pca = []
for eps in eps_grid_pca:
    for ms in min_samples_grid:
        db = DBSCAN(eps=eps, min_samples=ms).fit(df_pca)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1:
            results_dbscan_pca.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "score": silhouette_score(df_pca, labels)
            })

if results_dbscan_pca:
    df_db_pca = pd.DataFrame(results_dbscan_pca).sort_values(by='score', ascending=False)
    print("\n[Grid Search DF] DBSCAN (With PCA) Results:")
    print(df_db_pca.head())

    best_db_pca = df_db_pca.iloc[0]
    dbscan_pca = DBSCAN(eps=best_db_pca['eps'], min_samples=int(best_db_pca['min_samples'])).fit(df_pca)
    print(f"-> Best DBSCAN (With PCA) Score: {best_db_pca['score']:.4f}")
    final_results.append(
        {'Algorithm': 'DBSCAN', 'Data Type': 'With PCA (3D)', 'Silhouette Score': best_db_pca['score']})
    plot_pca_clusters(df_pca, dbscan_pca.labels_,
                      f"DBSCAN With PCA (eps={best_db_pca['eps']}, min={best_db_pca['min_samples']})")
    plot_world_map(df, dbscan_pca.labels_, f"World Map: DBSCAN With PCA")

# Model without PCA
results_dbscan_raw = []
for eps in eps_grid_raw:
    for ms in min_samples_grid:
        db = DBSCAN(eps=eps, min_samples=ms).fit(df_raw)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1:
            results_dbscan_raw.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "score": silhouette_score(df_raw, labels)
            })

if results_dbscan_raw:
    df_db_raw = pd.DataFrame(results_dbscan_raw).sort_values(by='score', ascending=False)
    print("\n[Grid Search DF] DBSCAN (Without PCA) Results:")
    print(df_db_raw.head())

    best_db_raw = df_db_raw.iloc[0]
    dbscan_raw = DBSCAN(eps=best_db_raw['eps'], min_samples=int(best_db_raw['min_samples'])).fit(df_raw)
    print(f"-> Best DBSCAN (Without PCA) Score: {best_db_raw['score']:.4f}")
    final_results.append(
        {'Algorithm': 'DBSCAN', 'Data Type': 'Without PCA (9D)', 'Silhouette Score': best_db_raw['score']})
    plot_pca_clusters(df_pca, dbscan_raw.labels_,
                      f"DBSCAN Without PCA (eps={best_db_raw['eps']}, min={best_db_raw['min_samples']})")
    plot_world_map(df, dbscan_raw.labels_, f"World Map: DBSCAN Without PCA")

# ==========================================
# 6. HDBSCAN CLUSTERING (PCA vs RAW)
# ==========================================
print("\n--- 3. HDBSCAN EXPERIMENTS ---")
min_cluster_grid = [3, 4, 5, 6, 7, 10]
min_sample_grid_hdb = [3, 4, 5, 6, 7, None]

# Model with PCA
results_hdbscan_pca = []
for mc in min_cluster_grid:
    for ms in min_sample_grid_hdb:
        hdb = HDBSCAN(min_cluster_size=mc, min_samples=ms).fit(df_pca)
        labels = hdb.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1:
            results_hdbscan_pca.append({
                "min_cluster": mc,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "score": silhouette_score(df_pca, labels)
            })

if results_hdbscan_pca:
    df_hdb_pca = pd.DataFrame(results_hdbscan_pca).sort_values(by='score', ascending=False)
    print("\n[Grid Search DF] HDBSCAN (With PCA) Results:")
    print(df_hdb_pca.head())

    best_hdb_pca = df_hdb_pca.iloc[0]
    best_ms_pca = None if pd.isna(best_hdb_pca['min_samples']) else int(best_hdb_pca['min_samples'])

    hdbscan_pca = HDBSCAN(min_cluster_size=int(best_hdb_pca['min_cluster']), min_samples=best_ms_pca).fit(df_pca)
    print(f"-> Best HDBSCAN (With PCA) Score: {best_hdb_pca['score']:.4f}")
    final_results.append(
        {'Algorithm': 'HDBSCAN', 'Data Type': 'With PCA (3D)', 'Silhouette Score': best_hdb_pca['score']})
    plot_pca_clusters(df_pca, hdbscan_pca.labels_, f"HDBSCAN With PCA")
    plot_world_map(df, hdbscan_pca.labels_, f"World Map: HDBSCAN With PCA")

# Model without PCA
results_hdbscan_raw = []
for mc in min_cluster_grid:
    for ms in min_sample_grid_hdb:
        hdb = HDBSCAN(min_cluster_size=mc, min_samples=ms).fit(df_raw)
        labels = hdb.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1:
            results_hdbscan_raw.append({
                "min_cluster": mc,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "score": silhouette_score(df_raw, labels)
            })

if results_hdbscan_raw:
    df_hdb_raw = pd.DataFrame(results_hdbscan_raw).sort_values(by='score', ascending=False)
    print("\n[Grid Search DF] HDBSCAN (Without PCA) Results:")
    print(df_hdb_raw.head())

    best_hdb_raw = df_hdb_raw.iloc[0]
    best_ms_raw = None if pd.isna(best_hdb_raw['min_samples']) else int(best_hdb_raw['min_samples'])

    hdbscan_raw = HDBSCAN(min_cluster_size=int(best_hdb_raw['min_cluster']), min_samples=best_ms_raw).fit(df_raw)
    print(f"-> Best HDBSCAN (Without PCA) Score: {best_hdb_raw['score']:.4f}")
    final_results.append(
        {'Algorithm': 'HDBSCAN', 'Data Type': 'Without PCA (9D)', 'Silhouette Score': best_hdb_raw['score']})
    plot_pca_clusters(df_pca, hdbscan_raw.labels_, f"HDBSCAN Without PCA")
    plot_world_map(df, hdbscan_raw.labels_, f"World Map: HDBSCAN Without PCA")

# ==========================================
# 7. AGGLOMERATIVE CLUSTERING (PCA vs RAW)
# ==========================================
print("\n--- 4. AGGLOMERATIVE EXPERIMENTS (K=3) ---")

plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(df_pca, method="ward"))
plt.title("Hierarchical Clustering Dendrogram (PCA Data)")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()

# --- LINKAGE TEST LOOP ---
print("\n[Linkage Test] Testing different linkage methods (With PCA):")
for link in ['ward', 'complete', 'average', 'single']:
    hc_test = AgglomerativeClustering(n_clusters=3, linkage=link).fit(df_pca)
    n_clusters_hc = len(set(hc_test.labels_))
    if n_clusters_hc > 1:
        print(f"-> Linkage: {link:8} | Silhouette Score: {silhouette_score(df_pca, hc_test.labels_):.4f}")
print("-" * 50)
# ---------------------------------------------

# Model with PCA
hc_pca = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(df_pca)
score_hc_pca = silhouette_score(df_pca, hc_pca.labels_)
print(f"-> Agglomerative (With PCA) Silhouette Score: {score_hc_pca:.4f}")
final_results.append({'Algorithm': 'Agglomerative', 'Data Type': 'With PCA (3D)', 'Silhouette Score': score_hc_pca})
plot_pca_clusters(df_pca, hc_pca.labels_, "Agglomerative Clustering With PCA (Ward, K=3)")
plot_world_map(df, hc_pca.labels_, "World Map: Agglomerative With PCA")

# Model without PCA
hc_raw = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(df_raw)
score_hc_raw = silhouette_score(df_raw, hc_raw.labels_)
print(f"-> Agglomerative (Without PCA) Silhouette Score: {score_hc_raw:.4f}")
final_results.append({'Algorithm': 'Agglomerative', 'Data Type': 'Without PCA (9D)', 'Silhouette Score': score_hc_raw})
plot_pca_clusters(df_pca, hc_raw.labels_, "Agglomerative Clustering Without PCA (Ward, K=3)")
plot_world_map(df, hc_raw.labels_, "World Map: Agglomerative Without PCA")

# ==========================================
# 8. FINAL COMPARISON
# ==========================================
print("\n" + "=" * 50)
print(" FINAL CLUSTERING COMPARISON RESULTS")
print("=" * 50)

df_compare = pd.DataFrame(final_results)
df_compare = df_compare.sort_values(by="Silhouette Score", ascending=False).reset_index(drop=True)
print(df_compare.to_string())

plt.figure(figsize=(12, 6))
sns.barplot(data=df_compare, x="Algorithm", y="Silhouette Score", hue="Data Type", palette="mako")
plt.title("Model Performance Comparison: With PCA vs Without PCA", fontsize=16, fontweight='bold')
plt.xlabel("Clustering Algorithm", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.ylim(0, df_compare["Silhouette Score"].max() + 0.1)
plt.legend(title="Data Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

