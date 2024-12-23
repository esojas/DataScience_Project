import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
file_path = '/content/updated_file - updated_file.csv.csv'
new_df = pd.read_csv(file_path)

# Prepare the dataset for clustering (remove non-numeric and target columns)
clustering_data = new_df.drop(columns=['state', 'num_of_obese'])

# Standardize the features
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(2, 11)  # Testing for 2 to 10 clusters
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.show()

# Using silhouette score to evaluate cluster quality
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(clustering_data_scaled)
    silhouette_scores.append(silhouette_score(clustering_data_scaled, labels))

# Plot the Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score for Different Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Choose the optimal number of clusters based on the above plots (e.g., k=3 for simplicity)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(clustering_data_scaled)

# Add the cluster labels back to the original dataset
new_df['Cluster'] = cluster_labels

# Visualize clusters using PCA (2D projection)
pca = PCA(n_components=2)
clustering_data_pca = pca.fit_transform(clustering_data_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=clustering_data_pca[:, 0], y=clustering_data_pca[:, 1], hue=cluster_labels, palette='Set2', s=100)
plt.title('Clusters Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Create a cluster summary to explain each cluster
cluster_summary = new_df.groupby('Cluster').agg({
    'num_of_obese': 'sum',
    'education': 'mean',
    'income': 'mean',
    'unemployment': 'mean',
    'num_of_health_clubs': 'mean',
    'num_exercise': 'mean',
    'num_not_exercise': 'mean',
    'restaurant_count': 'mean',
    'Cluster': 'count'
}).rename(columns={'Cluster': 'Count'})

# Save the cluster-enhanced dataset and summary
new_df.to_csv('Clustered_Dataset.csv', index=False)
cluster_summary.to_csv('Cluster_Summary.csv', index=True)

print("Clustered dataset saved as 'Clustered_Dataset.csv'")
print("Cluster summary saved as 'Kmean_Cluster_Summary.csv'")
