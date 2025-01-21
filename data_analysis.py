import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

data = data.dropna(subset=['CustomerID'])
data = data.drop(columns=['Description'])
print(data.isnull().sum())

data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
numeric_features = ['Quantity', 'UnitPrice', 'TotalPrice']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

scaled_data = data[['CustomerID', 'Quantity', 'UnitPrice', 'TotalPrice']].drop_duplicates(subset='CustomerID')
scaled_data = scaled_data.set_index('CustomerID')

silhouette = []
inertia = []
K = range(2,20)
for n in K:
    model = KMeans(n_clusters=n, random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(scaled_data, model.labels_))

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

model = KMeans(n_clusters=7,random_state=42)
model.fit(scaled_data)
print(np.unique(model.labels_))
print(f"Silhouette score: {silhouette_score(scaled_data, model.labels_)}")
scaled_data["Labels"] = model.labels_
print(scaled_data.head(10))
print(scaled_data.groupby("Labels").mean())


pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_data)
map = "tab20"

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=scaled_data['Labels'], cmap = map, s=50, alpha=0.7)
plt.title('K-Means Clustering (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.savefig(r"D:\Project\kmeans\visualisations\PCA.png")