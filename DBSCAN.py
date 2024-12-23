import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Load the dataset
file_path = '/content/updated_file - updated_file.csv.csv'
new_df = pd.read_csv(file_path)

# Prepare the dataset for clustering (remove non-numeric and target columns)
clustering_data = new_df.drop(columns=['state', 'num_of_obese'])

# Standardize the features
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Find the best parameters for DBSCAN using Silhouette Score
eps_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples_values = [3, 5, 10]
silhouette_results = []

best_eps = None
best_min_samples = None
best_silhouette = -1
best_labels = None

# Test different combinations of eps and min_samples
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(clustering_data_scaled)
        if len(set(labels)) > 1:  # Silhouette score is undefined if there's only one cluster
            score = silhouette_score(clustering_data_scaled, labels)
            silhouette_results.append((eps, min_samples, score))
            if score > best_silhouette:
                best_silhouette = score
                best_eps = eps
                best_min_samples = min_samples
                best_labels = labels
        else:
            silhouette_results.append((eps, min_samples, -1))  # Indicate invalid clustering

print(f"Best Silhouette Score: {best_silhouette}")
print(f"Optimal eps: {best_eps}, Optimal min_samples: {best_min_samples}")

# Plot the Silhouette Scores
silhouette_df = pd.DataFrame(silhouette_results, columns=['eps', 'min_samples', 'silhouette_score'])
# Create a pivot table for the heatmap
pivot_table = silhouette_df.pivot(index="eps", columns="min_samples", values="silhouette_score")

# Plot the Silhouette Scores as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Silhouette Score'})
plt.title('Silhouette Score for Different DBSCAN Parameters')
plt.xlabel('min_samples')
plt.ylabel('eps')
plt.tight_layout()
plt.show()

# Apply DBSCAN with the optimal parameters
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
cluster_labels = dbscan.fit_predict(clustering_data_scaled)

# Add the cluster labels back to the original dataset
new_df['Cluster'] = cluster_labels

# Save the clustered dataset to a CSV file
new_df.to_csv('Clustered_Dataset_DBSCAN.csv', index=False)
print("Clustered dataset saved as 'Clustered_Dataset_DBSCAN.csv'")

# Visualize clusters using PCA (2D projection)
pca = PCA(n_components=2)
clustering_data_pca = pca.fit_transform(clustering_data_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=clustering_data_pca[:, 0], y=clustering_data_pca[:, 1], hue=cluster_labels, palette='Set2', s=100)
plt.title('Clusters Visualization (DBSCAN)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Filter numeric columns for the summary table
numeric_columns = new_df.select_dtypes(include=['number'])

# Group the data by clusters and calculate the mean for numeric columns
cluster_summary = numeric_columns.groupby(new_df['Cluster']).mean()

# Include a count of data points in each cluster
cluster_summary['Count'] = new_df['Cluster'].value_counts()

# Save the summary table to a CSV file
cluster_summary.to_csv('Cluster_Summary_DBSCAN.csv')
print("Cluster Summary Table:")
print(cluster_summary)
