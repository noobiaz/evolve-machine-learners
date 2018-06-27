import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import math
from sklearn.cluster import KMeans

def generate_train_test_dataset():
    # Generate dataset
    X, y = make_blobs(centers=3, n_samples=500, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    return X_train, y_train, X_test, y_test


def visualize_data(X):
    # Visualize
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')


def plot_result(X, centroids, classes):
    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in classes]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$');


def train(x_train, k):
    kmeans = KMeans(n_clusters=k, random_state=1, init='random')
    kmeans.fit(x_train)
    return kmeans


def predict(kmeans, data):
    result=kmeans.predict(data)
    return result

x_train, y_train, x_test, y_test=generate_train_test_dataset()
k=3
kmeans=train(x_train, k)
y_train_hat=predict(kmeans, x_train)
y_test_hat=predict(kmeans, x_test)

plot_result(x_train, kmeans.cluster_centers_, y_train_hat)
plot_result(x_test, kmeans.cluster_centers_, y_test_hat)

plt.show()