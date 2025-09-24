import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

def plot_clusters(data, assignments=None, k=1, calculated=None, actual=None):
    if data.shape[1] > 2:
        print("Unable to plot >2D data")
        return 1
    plt.figure()
    if k == 1: # Plot data
        plt.scatter(data[:,0], data[:,1], alpha=0.3, label=f"Data")
    else: # Plot clusters
        for i in range(k):
            subset = data[assignments == i,:]
            plt.scatter(subset[:,0], subset[:,1], alpha=0.5, label=f"Cluster {i}")
    if calculated is not None:
        plt.scatter(calculated[:,0], calculated[:,1], c='r', label="Calculated")
    if actual is not None:
        plt.scatter(actual[:,0], actual[:,1], c='k', label="Actual")
    plt.legend()
    plt.show()

def cupy_generate_data(k=3, n=10, d=2, cmin=0, cmax=10):
    '''
    Generate a random dataset for clustering using cupy
    k - number of clusters
    n - number of data points
    d - dimension of data points
    '''
    C = cp.random.randint(cmin, cmax, size=(k,d))
    D = cp.random.randn(n, d)+C[cp.random.choice(k, n, replace=True)]
    return D, C

def plot_timing(numpy_xy, skl_xy, cuml_xy):
    '''
    Plots 3 lists of the structure [[N data points], [runtime]]
    '''
    plt.figure()
    plt.loglog(numpy_xy[0], numpy_xy[1], label="numpy")
    plt.loglog(skl_xy[0], skl_xy[1], label="sklearn")
    plt.loglog(cuml_xy[0], cuml_xy[1], label="cuML")
    plt.xlabel("# points")
    plt.ylabel("Seconds")
    plt.legend()
    plt.show()

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

import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import os

def mp_generate_data(k=3, n=10, d=2, cmin=0, cmax=10, NP=0):
    '''
    Generate a random dataset for clustering.
    k - number of clusters
    n - number of data points
    d - dimension of data points
    cmin - smallest centroid value
    cmax - largest centroid value
    '''
    C = np.random.randint(cmin, cmax, size=(k,d))
    global D
    
    D = RawArray('f', n*d) # fp32, n*d points
    #np_chromArray = np.frombuffer(chromArray, dtype='i')
    NP = 2 if n < 10000 else NP
    NP = os.cpu_count() if not NP else NP
    pool = mp.Pool(NP)
    pool.map(mp_data_worker, [(n, k, d, C, i, NP) for i in range(NP)])
    pool.close()
    pool.join()
    npD = np.frombuffer(D, 'f').reshape((n,d))
    return npD, C

def mp_data_worker(args):
    n, k, d, C, i, NP = args
    np.random.seed()
    global D
    np_D = np.frombuffer(D, dtype='f').reshape((n,d))
    block_size = n//NP
    sI = block_size*i
    eI = block_size+sI if i < (NP-1) else n
    block_size = eI-sI
    #print(f"Thread {i} editing [{sI}:{eI}]")
    np_D[sI:eI,:] = np.random.randn(block_size, d)+C[np.random.choice(k, block_size, replace=True)]
    return 0
