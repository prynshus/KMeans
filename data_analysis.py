import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

data = pd.read_excel(r"D:\Project\kmeans\Dataset\Online Retail.xlsx")
print(data.head())
print(data.describe())
print(data.info())

cat_columns = data.select_dtypes(include = "object").columns
num_columns = data.select_dtypes(include = ["int64","float64"]).columns
for cat in cat_columns:
    print(f"{cat}: ")
    print(pd.unique(data[cat]))

print(data.isnull().sum())

data["TotalCost"] = data["UnitPrice"] * data["Quantity"]
reference_date = data["InvoiceDate"].max()
customer_data_rfm = data.groupby("CustomerID").agg({
    "InvoiceNo" : "count",
    "TotalCost" : "sum",
    "InvoiceDate" : lambda x: (reference_date - x.max()).days
})
customer_data_rfm = customer_data_rfm.rename(columns={
    "InvoiceNo" : "frequency",
    "TotalCost" : "monatory",
    "InvoiceDate" : "recency"
}).reset_index()
print(customer_data_rfm.head())

scale = MinMaxScaler()
numeric_features = ['frequency', 'monatory', 'recency']
scaled = scale.fit_transform(customer_data_rfm[numeric_features])
customer_data_rfm_scaled = pd.DataFrame(data=scaled, columns=['frequency', 'monatory', 'recency'])
customer_data_rfm_scaled["CustomerID"] = customer_data_rfm["CustomerID"]
print(customer_data_rfm_scaled)

silhouette = []
inertia = []
K = range(2,20)
for n in K:
    model = KMeans(n_clusters=n, random_state=42)
    model.fit(customer_data_rfm_scaled)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(customer_data_rfm,model.labels_))

fig, axes = plt.subplots(figsize=(10,5),nrows=1,ncols=2)
axes[0].plot(K,inertia,'bo-')
axes[0].set_xlabel("Number of clusters")
axes[0].set_ylabel("inertia")
axes[0].set_title("elbow curve")

axes[1].plot(K,silhouette,'go-')
axes[1].set_xlabel("Number of clusters")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette analysis")
fig.savefig(r"D:\Project\kmeans\visualisations\curvePlot.png")

model = KMeans(n_clusters=8,random_state=42)
model.fit(customer_data_rfm_scaled)
print(np.unique(model.labels_))
customer_data_rfm["Labels"] = model.labels_
print(customer_data_rfm_scaled.head(10))
print(customer_data_rfm.groupby("Labels").mean())


tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(customer_data_rfm_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X_tsne[:,0], X_tsne[:,1], X_tsne[:,2],
    c=customer_data_rfm['Labels'], cmap='viridis', s=50, alpha=0.7
)
ax.set_title("3D Cluster Visualization")
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
fig.savefig(r"D:\Project\kmeans\visualisations\3DPlot.png")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(customer_data_rfm_scaled)
map = "tab20"

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customer_data_rfm['Labels'], cmap = map, s=50, alpha=0.7)
plt.title('K-Means Clustering (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.savefig(r"D:\Project\kmeans\visualisations\PCA.png")