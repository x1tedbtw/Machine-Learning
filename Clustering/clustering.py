import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import os


def check_structure(file_path):
    print(f"Checking structure of {file_path}:")
    with open(file_path, 'r') as file:
        for _ in range(5):
            print(file.readline().strip())
    print()


file_paths = ['s1.txt', 's2.txt', 's3.txt', 's4.txt']

for file_path in file_paths:
    if os.path.exists(file_path):
        check_structure(file_path)
    else:
        print(f"File '{file_path}' not found.")

def parse_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:

                x, y = map(float, line.strip().split())
                data.append((x, y))
            except ValueError:
                print(f"Skipping line '{line.strip()}' in file '{file_path}' as it cannot be parsed as floats.")
    return data


parsed_data = {}
for file_path in file_paths:
    if os.path.exists(file_path):
        parsed_data[file_path] = parse_data(file_path)


print("Parsed data for s1.txt:")
print(parsed_data['s1.txt'][:5])


def apply_kmeans(data, num_clusters, init_method='k-means++', max_iter=300):
    kmeans = KMeans(n_clusters=num_clusters, init=init_method, max_iter=max_iter)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    return labels, centroids

def visualize_clusters(data, labels, centroids, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', label='Centroids')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Load and prepare data
file_paths = ['s1.txt', 's2.txt', 's3.txt', 's4.txt']
datasets = {}
for file_path in file_paths:
    data = np.genfromtxt(file_path, dtype=float, delimiter=None, comments=None)
    datasets[file_path] = data


num_clusters = 3
init_methods = ['k-means++', 'random']
max_iters = [100, 300]

for file_path, data in datasets.items():
    print(f"Dataset: {file_path}")
    for init_method in init_methods:
        for max_iter in max_iters:
            try:
                labels, centroids = apply_kmeans(data, num_clusters, init_method, max_iter)
                silhouette = silhouette_score(data, labels)
                print(f"Init Method: {init_method}, Max Iter: {max_iter}, Silhouette Score: {silhouette}")
                visualize_clusters(data, labels, centroids, f"K-means Clustering (Init: {init_method}, Max Iter: {max_iter}) - {file_path}")
            except ValueError as e:
                print(f"Skipping file '{file_path}' with error: {e}")


def parse_spiral_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, cluster = map(float, line.strip().split())
            data.append((x, y, cluster))
    return np.array(data)

spiral_data = parse_spiral_data('spiral.txt')

features = spiral_data[:, :2]
true_labels = spiral_data[:, 2].astype(int)


num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_

# Visualize clustering results
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 1], c=predicted_labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', label='Centroids')
plt.title('K-means Clustering on Spiral Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Compare with ground truth
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 1], c=true_labels, cmap='viridis', alpha=0.5)
plt.title('Ground Truth Clusters on Spiral Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

