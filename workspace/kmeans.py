#!/usr/bin/env python

import numpy as np
#import cupy as np

def euclidean_distance(point1, point2):
    '''
    Calculates euclidean distance between two points with the same dimensions

    input:
    - point1 - N-D numpy array
    - point2 - N-D numpy array

    returns:
    - float - distance
    '''
    return np.linalg.norm(point1-point2, axis=0)

def assign_points_to_clusters(data, centroids):
    '''Assigns each point to the closest centroid'''
    # Create indices for pairwise comparisons
    Di, Ci = np.meshgrid(np.arange(data.shape[0]),np.arange(centroids.shape[0]))
    # Calculate pairwise distances and then return the index of the closest centroid
    return np.argmin(np.linalg.norm(data[Di,:]-centroids[Ci,:], axis=2), axis=0)
    #assignment = np.zeros(len(data))-1
    #for i in range(len(data)):
    #    assignment[i] = find_closest_centroid(data[i], centroids)
    #return assignment

def update_centroids(data, assignments, centroids):
    # Updates centroids based on cluster assignments
    new_centroids = np.zeros(centroids.shape)
    for i in range(len(centroids)):
        indices = assignments == i
        if np.sum(indices) > 0: # If cluster not empty
            new_centroids[i,:] = np.mean(data[indices], axis=0)
        else: # If cluster is empty
            new_centroids[i,:] = data[np.random.randint(len(data))]
    return new_centroids

def k_means(data, k, max_iterations=100):
    np.random.seed(0) # Set seed to repititions are the same
    #print(f"Making {k} clusters of {data.shape} data points")
    # Randomly select k data points as initial centroids
    centroids = data[np.random.choice(len(data), k, replace=False)]

    # Stop if max iterations reached
    for i in range(max_iterations):
        assignments = assign_points_to_clusters(data, centroids)
        new_centroids = update_centroids(data, assignments, centroids)
        
        # Check for convergence - using same tolerance as sklearn.cluster.KMeans
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            return assignments, new_centroids # Stop if centroids are stable
        centroids = new_centroids
        
    # Return final clusters and centroids
    return assignments, centroids

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

if __name__ == "__main__":
    from time import time
    k, d = 4, 4

    for N in [10**power for power in range(1,8)]:
        data, real_centroids = generate_data(k=k, n=N, d=d, cmin=0, cmax=100)
        start_time = time()
        # Time a single iteration because k-means is non-deterministic
        k_means(data, k, max_iterations=1)
        elapsed_time = time()-start_time
        print(f"Finished kmeans on {N} points in {np.round(elapsed_time,4)} seconds")
        del data, real_centroids
    print("Done") # I sometimes wait for another step
