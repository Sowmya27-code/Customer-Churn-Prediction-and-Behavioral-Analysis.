from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict

class ClusterAnalyzer:
    def __init__(self, max_clusters: int = 10):
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
    
    def find_optimal_clusters(self, X: pd.DataFrame) -> Dict:
        inertias = []
        silhouette_scores = []
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute metrics for different cluster numbers
        for k in range(2, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot elbow curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(2, self.max_clusters + 1), inertias, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.savefig("visualizations/elbow_curve.png")
        plt.close()
        
        return {
            "optimal_k": silhouette_scores.index(max(silhouette_scores)) + 2,
            "silhouette_scores": dict(enumerate(silhouette_scores, 2)),
            "inertias": dict(enumerate(inertias, 2))
        }