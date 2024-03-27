from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data,k):
    sse_normal = []
    sse_inertia = []
    ss = StandardScaler()
    data = ss.fit_transform(data)
    for clusters_num in range(1, k + 1):
        kmeans = KMeans(n_clusters=clusters_num)
        kmeans.fit(data)
        sse = 0
        for i in range(clusters_num):
            cluster_points = data[kmeans.labels_ == i]
            centroid = kmeans.cluster_centers_[i]
            sse += np.sum((cluster_points - centroid) ** 2)
        sse_normal.append(sse)
        
        # Calculate inertia
        sse_inertia.append(kmeans.inertia_)
    
    return sse_normal, sse_inertia
    



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    
    data,labels = make_blobs(n_samples=20,centers=5,random_state=12,center_box=(-20.0,20.0))
    dct = answers["2A: blob"] = [np.array(data[:,0]),np.array(data[:,1]),np.array(labels)]
    #print(dct)
    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans
    sse_B,sse_D = fit_kmeans(data,8)
    # print(sse_B)
    # print(sse_D)
    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = [(index, value) for index, value in enumerate(sse_B)]
    plt.figure(figsize=(10, 6))
    plt.plot([1,2,3,4,5,6,7,8], sse_B, marker='o', linestyle='-')
    plt.title('K Value vs SSE')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.xticks([1,2,3,4,5,6,7,8])
    plt.grid(True)
    plt.savefig('2C.jpg')

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = [(index, value) for index, value in enumerate(sse_D)]
    plt.figure(figsize=(10, 6))
    plt.plot([1,2,3,4,5,6,7,8], sse_D, marker='o', linestyle='-')
    plt.title('K Value vs SSE(Inertia)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE (Sum of Squared Errors) Inertia')
    plt.xticks([1,2,3,4,5,6,7,8])
    plt.grid(True)
    plt.savefig('2D.jpg')

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
