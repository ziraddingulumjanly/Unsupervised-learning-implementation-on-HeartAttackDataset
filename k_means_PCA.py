# Libraries needed
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
k_values = range(1, 21)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method for Optimal Cluster Number', fontsize=14)
plt.grid()
plt.show()   # t-SNE

# Using 3 clusters for K-Means
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Adding cluster labels to the dataset
unsupervised_data['cluster'] = clusters+1

# Reducing data to 2D using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Creating a DataFrame for PCA results and clusters
pca_df = pd.DataFrame(pca_data, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Cluster'] = clusters+1

# Scatter plot of PCA results with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='Principal Component 1', y='Principal Component 2', hue='Cluster', palette='tab10', s=100)
plt.title('PCA Visualization of Clusters', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
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


