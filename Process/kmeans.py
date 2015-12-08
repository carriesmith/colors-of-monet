from sklearn.utils import check_random_state
import numpy as np

def _distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def _max_min_dist_centroid(centroids, X):  
    n_clusters = centroids.shape[0]
    n_sample = X.shape[0]

    min_dist = np.empty((n_sample, 1))
    min_dist.fill(np.infty)
    
    # Compute Euclidean distance between each point and each of
    # the centroids.
    for xi in range(n_sample):
        for ci in range(n_clusters):
            # If the Euclidean distance between this centroid and the data
            #  point is less than the previous minimum distance, then save it!
            min_dist[xi] = min(min_dist[xi], _distance(X[xi], centroids[ci]))
    
    # Return the data point with the largest minimum distance
    return X[ min_dist.argmax() ]

def _init_centroids(X, k, random_seed=None):
    
    n_sample = X.shape[0]
    n_features = X.shape[1]
    
    centroids = np.zeros((k, n_features))
    
    # The first centroid chosen at random
    random_state = check_random_state(random_seed)
    centroids[0] = X[random_state.randint(X.shape[0])]
    
    # Add a new centroid each iteration
    # the data point with the largest minimum distance to previously
    # selected centroids.
    for centi in range(1,k):
        centroids[centi] = _max_min_dist_centroid(centroids[:centi], X)
        
    return centroids

def _assign_cluster(X, centroids):
    
    n_sample = X.shape[0]
    n_clusters = centroids.shape[0]
    
    labels = -np.ones((n_sample), dtype = int)
    WCSS = 0
    
    for xi in range(n_sample):
        min_dist = np.infty
        for ci in range(n_clusters):
            dist = _distance(X[xi], centroids[ci])
            if (dist < min_dist):
                labels[xi] = ci
                min_dist = dist
        # sum of the euclidean distances to nearest centroid        
        WCSS += min_dist
    
    return labels, WCSS

def _update_clusters(X, labels, k):
    
    n_sample = X.shape[0]
    n_features = X.shape[1]
    
    sum_data = np.zeros((k,n_features))
    count = np.zeros(k)
    
    for xi in range(n_sample):
        sum_data[labels[xi]] += X[xi]
        count[labels[xi]] += 1
    
    for ci in range(k):
        sum_data[ci] /= count[ci]

    return sum_data

def _kmeans_once(X, k, maxiter=1000, random_seed=None, verbose=False):
    
    centroids = _init_centroids(X,k,random_seed)
    labels, WCSS = _assign_cluster(X, centroids)
    
    for iter in range(maxiter):
            
        new_centroids = _update_clusters(X, labels, k)
        labels, WCSS = _assign_cluster(X, new_centroids)
        
        if (verbose): print new_centroids
        
        if (new_centroids == centroids).all():
            break
        
        centroids = new_centroids
    
    if (verbose): 
        print "Converged in ", iter, "iterations"
        print centroids
    
    return labels, centroids, WCSS

def kmeans( X, k, n_init=10, maxiter=1000, random_seed=None, verbose = False ):
    
    n_sample = X.shape[0]
    n_features = X.shape[1]
    
    best_WCSS = np.infty
    best_labels = -np.ones((n_sample), dtype = int)
    best_centroids = np.zeros((k, n_features))
    
    for iter in range(n_init):
        labels, centroids, WCSS = _kmeans_once(X, k, maxiter=maxiter, random_seed=random_seed)
        if (verbose):
            print "Iteration ", iter, " WCSS = ", WCSS
            print centroids
        if (WCSS < best_WCSS):
            best_labels, best_centroids, best_WCSS = labels, centroids, WCSS
    
    return best_labels, best_centroids, best_WCSS