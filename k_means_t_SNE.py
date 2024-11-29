# Libraries needed
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
heart_attack_data = pd.read_csv('Heart Attack.csv')

# Dropping the 'class' column for unsupervised learning
unsupervised_data = heart_attack_data.drop(columns=['class'])

# Scaling the numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(unsupervised_data)

# Applying K-Means Clustering
inertia = []
silhouette_scores = []
k_values = range(2, 41)  

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# Plotting the Elbow Curve with Silhouette Scores
fig, ax1 = plt.subplots(figsize=(10, 6))

# Elbow curve
ax1.plot(k_values, inertia, marker='o', linestyle='--', label='Inertia')
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Method and Silhouette Score', fontsize=14)
ax1.grid()
ax1.legend(loc='upper right')

# Silhouette scores
ax2 = ax1.twinx()
ax2.plot(k_values, silhouette_scores, marker='s', linestyle='-', color='orange', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.legend(loc='lower right')

plt.show()

# Using 8 clusters for K-Means
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Adding cluster labels to the dataset
unsupervised_data['cluster'] = clusters+1

# Imlementation of t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter = 1000)
tsne_data = tsne.fit_transform(scaled_data)

tsne_df = pd.DataFrame(tsne_data, columns=['TSNE Component 1', 'TSNE Component 2'])
tsne_df['Cluster'] = clusters + 1


# Scatter plot of t-SNE results with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tsne_df, x='TSNE Component 1', y='TSNE Component 2', hue='Cluster', palette='tab10', s=100)
plt.title('t-SNE Visualization of Clusters', fontsize=14)
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.legend(title='Cluster', fontsize=10)
plt.grid()
plt.show()

# Feature distributions by cluster
features_to_plot = ['age', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']

for feature in features_to_plot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=unsupervised_data, x='cluster', y=feature, palette='Set2')
    plt.title(f'{feature.capitalize()} Distribution Across Clusters', fontsize=14)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel(feature.capitalize(), fontsize=12)
    plt.grid()
    plt.show()


