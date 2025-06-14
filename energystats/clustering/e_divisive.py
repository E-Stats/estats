import numpy as np

class EDivisive:
    def __init__(self, X :np.ndarray, p0 :float = 0.05, max_clusters :int = 2, alpha :float = 1.0):

        """
        Initializes the E-Divisive time series clusering model.
        
        Args:
            X (np.ndarray): Data points.
            p0 (float): Clustering tolerance, maxmimum p-value to partition, functions as a stopping criterion.
            max_clusters (int): Maximum number of clusters to form, functions as another stopping criterion.
            alpha (float): Exponent for distance calculation, default is 1.0.
        """
        self.X = X
        self.p0 = p0
        self.max_clusters = max_clusters
        self.labels_ = None
        self.cluster_centers_ = None
    def divide(self, l, r):
        
    def fit(self):
        """
           Fits the model
           Args:
               None
           Returns:
               A ClusterResults object containing the clusters
        """

        