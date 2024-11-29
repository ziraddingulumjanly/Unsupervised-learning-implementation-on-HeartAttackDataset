# Import necessary libraries
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset 
file_path = 'Heart Attack.csv'
heart_attack_data = pd.read_csv(file_path)

# Step 1: Preprocessing - Drop categorical columns and scale numeric data
data_for_clustering = heart_attack_data.drop(columns=['class'], errors='ignore')  # Ignore if 'class' doesn't exist
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

# Step 2: Perform Hierarchical Clustering
# Using Ward's linkage method
linkage_matrix = linkage(scaled_data, method='ward')

# Step 3: Plot the Dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance')
plt.axhline(y=25, color='r', linestyle='--', label='Cut-off Threshold')  # Threshold for clusters
plt.legend()
plt.show()

# Step 4: Assign Clusters Using the Threshold
threshold = 25  # (recommended) can be adjusted
clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

# Step 5: Add Cluster Labels to the Data
heart_attack_data['Cluster'] = clusters

# Step 6: Summarize the Clusters
cluster_summary = heart_attack_data.groupby('Cluster').agg(
    Count=('Cluster', 'size'),
    Mean_Age=('age', 'mean'),
    Mean_Glucose=('glucose', 'mean'),
    Mean_Troponin=('troponin', 'mean'),
    Mean_PressureHigh=('pressurehight', 'mean'),
    Mean_PressureLow=('pressurelow', 'mean')
).sort_values(by='Count', ascending=False)

# Display the summary
print("Cluster Summary:")
print(cluster_summary)
