import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import Birch
from sklearn.decomposition import PCA

# Load the dataset
file_path = '/content/updated_file - updated_file.csv.csv'
new_df = pd.read_csv(file_path)

# Prepare the dataset for clustering (remove non-numeric and target columns)
clustering_data = new_df.drop(columns=['state', 'num_of_obese'])

# Standardize the features
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
k_values = range(2, 11)  # Testing for 2 to 10 clusters

for k in k_values:
    birch = Birch(n_clusters=k)
    labels = birch.fit_predict(clustering_data_scaled)
    silhouette_scores.append(silhouette_score(clustering_data_scaled, labels))

# Plot the Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', color='blue')
plt.title('Silhouette Score for Different Clusters (BIRCH)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Choose the optimal number of clusters based on the above plots (e.g., k=3 for simplicity)
optimal_k = 2
birch = Birch(n_clusters=optimal_k)
cluster_labels = birch.fit_predict(clustering_data_scaled)

# Add the cluster labels back to the original dataset
new_df['Cluster'] = cluster_labels

# Visualize clusters using PCA (2D projection)
pca = PCA(n_components=2)
clustering_data_pca = pca.fit_transform(clustering_data_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=clustering_data_pca[:, 0], y=clustering_data_pca[:, 1], hue=cluster_labels, palette='Set2', s=100)
plt.title('Clusters Visualization (PCA) - BIRCH')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Create a summary table for the clusters
# Filter numeric columns for the summary table
numeric_columns = new_df.select_dtypes(include=['number'])

# Group the data by clusters and calculate the mean for numeric columns
cluster_summary = numeric_columns.groupby(new_df['Cluster']).mean()

# Include a count of data points in each cluster
cluster_summary['Count'] = new_df['Cluster'].value_counts()

# Save the summary table to a CSV file
cluster_summary.to_csv('Cluster_Summary_BIRCH.csv')
print("Cluster Summary Table:")
print(cluster_summary)

# Save the cluster-enhanced dataset
new_df.to_csv('Clustered_Dataset_BIRCH.csv', index=False)
print("Clustered dataset saved as 'Clustered_Dataset_BIRCH.csv'")
