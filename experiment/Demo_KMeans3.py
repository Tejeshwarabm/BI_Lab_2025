# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Step 2: Create a simple dataset
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [19, 21, 20, 23, 31, 45, 40, 38, 42, 50],
    'Annual Income (LPA)': [15, 15, 16, 16, 28, 36, 50, 60, 65, 70],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 40, 65, 50, 30]
}
df = pd.DataFrame(data)

# Step 3: Select features for clustering
X = df[['Annual Income (LPA)', 'Spending Score (1-100)']]

# Step 4: Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Step 5: Visualize the clusters
plt.scatter(X['Annual Income (LPA)'], X['Spending Score (1-100)'],
            c=df['Cluster'], cmap='rainbow', s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            color='black', marker='X', s=200, label='Centroids')
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (LPA)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()

# Step 6: View clustered data
print(df)
