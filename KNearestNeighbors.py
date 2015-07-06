import numpy as np
from collections import Counter
from metric_distance import euclidean_distance, cosine_distance

class KNearestNeighbors(object):
    def __init__(self, k=5, distance='euclidean'):
        # number of neighbors who vote
        self.k = k
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

    def vote_from_distance(self, d):
        '''
        INPUT:
            - d: 1d np array of distancs
        OUTPUT:
            - float of classification
        '''
        mask = np.argsort(d)[0:self.k]
        votes = self.y_fit[mask]
        counts = Counter(votes)
        return count.most_common(1)[0][0]

    def predict(self, X_pred):
        '''
        INPUT:
            - X_pred: 2d np array of features
        '''

        return np.array([self.vote_from_distance(
            self.distance_metric(m, self.X_fit)) for m in X_pred])


