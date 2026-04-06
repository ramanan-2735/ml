import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

image = load_sample_image("china.jpg")
image = np.array(image)

X = image.reshape(-1, 3)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

segmented_image = centroids[labels].reshape(image.shape).astype(np.uint8)


plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(segmented_image)
plt.title(f"Segmented Image (K={k})")
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow([centroids.astype(np.uint8)])
plt.title("Cluster Centroids (Colors)")
plt.axis('off')

plt.tight_layout()
plt.show()