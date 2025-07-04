import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

# For this example, use 'Annual Income (k$)' and 'Spending Score (1-100)'
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: reduce to 2D with PCA if more dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 2. Fit K-Means with an initial K
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 3. Elbow Method to find optimal K
wcss = []  # within-cluster sum of squares
for i in range(1, 11):
    kmeans_i = KMeans(n_clusters=i, random_state=42)
    kmeans_i.fit(X_scaled)
    wcss.append(kmeans_i.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()

# 4. Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='black', marker='X')
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()

# 5. Evaluate using Silhouette Score
score = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score: {score:.2f}")
