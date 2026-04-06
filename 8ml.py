import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
normal_data = np.random.randn(300, 2)
anomalies = np.random.uniform(low=-6, high=6, size=(30, 2))
X = np.vstack([normal_data, anomalies])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
noise = labels == -1
clusters = labels != -1
plt.figure()
plt.scatter(X_scaled[clusters, 0], X_scaled[clusters, 1], label="Normal Data")
plt.scatter(X_scaled[noise, 0], X_scaled[noise, 1], label="Anomalies (Outliers)")
plt.title("DBSCAN Outlier Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
print("Total Data Points:", len(X))
print("Detected Outliers:", np.sum(noise))
print("Clusters Found:", len(set(labels)) - (1 if -1 in labels else 0))