####################################################################
# Program-5: Cluster analysis using k-means algorithm for a given  #
# customer data set (use Python).                                  #
#                                                                  #
####################################################################

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# initialize number of customer
n_customer = 100

# Step 2: Create a simple dataset
data = {
    'CustomerID': np.arange(1, n_customer+1),
    'Age': np.random.randint(18, 60, n_customer),
    'Annual Income (LPA)': np.random.randint(3, 45, n_customer),
    'Spending Score (1-100)': np.random.randint(1, 100, n_customer)
}
df = pd.DataFrame(data)

# Step 3: Select features for clustering
X = df[['Annual Income (LPA)', 'Spending Score (1-100)']]

# Step 4: Fit and Transform input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Find Silhouette Score for n-cluster
wcss_KM = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss = kmeans.inertia_      # Find "Within Cluster Square Sum"
    wcss_KM.append(wcss)

    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Step 6: Plot elbow curve
plt.figure(figsize=(15, 5))

# Elbow method plot
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss_KM, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Step 7: Silhouette score plot
plt.subplot(1, 2, 2)
plt.plot(range(2, 10), silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k values')
plt.grid(True)

plt.tight_layout()
plt.show()