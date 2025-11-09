
# Stage 1: Import Libraries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

##------------------------------------------------------------------------------------------------
## Stage 2: Create Sample DATA
# Create a dataset with 3 clusters
X, y = make_blobs(n_samples=200, centers=3, random_state=42)

##------------------------------------------------------------------------------------------------
## Stage 3: Apply K-Means
# Create KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit and predict cluster labels
y_pred = kmeans.fit_predict(X)

##------------------------------------------------------------------------------------------------
## Stage 4: Visualize the Clusters
# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow', s=50)

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='black', marker='X', s=200, label='Centroids')

plt.title("K-Means Clustering Example")
plt.legend()
plt.show()
