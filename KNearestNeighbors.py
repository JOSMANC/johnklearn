import numpy as np
from collections import Counter
from metric_distance import euclidean_distance, cosine_distance

class KNearestNeighbors(object):
    def __init__(self, k=5, distance='euclidean'):
        # number of neighbors who vote
        self.neighbors = k
        # distance metric to consider
        if distance == 'euclidean':
            self.distance_metric = euclidean_distance
        elif distance == 'cosine':
            self.distance_metric = cosine_distance
        # values to fit
        self.X_fit = None
        self.y_fit = None

    def fit(self, X, y):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
        '''
        self.X_fit = X
        self.y_fit = y

    def predict(self, x):
        '''
        INPUT:
            - x: 2d np array of features
        OUTPUT:
            - list of classifications
        '''
        return np.apply_along_axis(self._vote_from_distance, 1, x)
        
    def _vote_from_distance(self, m):
        '''
        INPUT:
            - m: 1d np array of numeric measurements
        OUTPUT:
            - float of classification
        '''
        d = self.distance_metric(x, self.X_fit)
        mask = np.argsort(d)[0:self.neighbors]
        votes = self.y_fit[mask]
        counts = Counter(votes)
        return count.most_common(1)[0][0]



