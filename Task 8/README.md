# K-Means Clustering: Customer Segmentation

## What this covers
This project:
1. Loads a customer dataset.
2. Uses K-Means to segment customers based on features.
3. Applies the Elbow Method to find the best number of clusters.
4. Visualizes the clusters.
5. Evaluates clustering quality using the Silhouette Score.

## Dataset
- `Mall_Customers.csv` (commonly used for clustering exercises)

## Key Steps

**Preprocessing**  
- Two features: Annual Income and Spending Score  
- Scaled with `StandardScaler` for better clustering performance

**Modeling**  
- K-Means with different values of K
- Elbow Method plot helps choose optimal K
- Cluster assignment and visualization

**Evaluation**  
- Silhouette Score measures cluster separation and cohesion

## Run

```bash
python kmeans_clustering.py
