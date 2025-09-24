#!/usr/bin/env python

from sklearn.cluster import KMeans
from time import time
import numpy as np

k, d = 4, 4

def generate_data(k=3, n=10, d=2, cmin=0, cmax=10):
    '''
    Generate a random dataset for clustering.
    k - number of clusters
    n - number of data points
    d - dimension of data points
    cmin - smallest centroid value
    cmax - largest centroid value
    '''
    # Generate random integers for centroids
    C = np.random.randint(cmin, cmax, size=(k,d))
    # Create data points by adding random noise to centroids
    D = np.random.randn(n, d)+C[np.random.choice(k, n, replace=True)]
    return D, C

for N in [10**power for power in range(1,8)]:
    data, real_centroids = generate_data(k=k, n=N, d=d, cmin=0, cmax=100)
    start_time = time()
    assignments = KMeans(n_clusters=k, random_state=0, max_iter=1).fit_predict(data)
    elapsed_time = time()-start_time
    print(f"Finished sklearn-kmeans on {N} points in {np.round(elapsed_time,4)} seconds")
    del data, real_centroids
print("Done")