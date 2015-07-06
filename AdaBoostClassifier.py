import numpy as np
from ClassificationDecisionTree import ClassificationDecisionTree as DT


class AdaBoostClassifier(object):
    '''
    An adaptive boosting using decision trees for binary classification
    '''
    def __init__(self, n_estimators=50, learn_rate=1.):

        # number of decision trees to create
        self.n_trees = n_estimators
        # rate with which esitmators are considered
        # where lower = slower but with likely more accuracy
        self.learn_rate = learn_rate
        # array to store trees
        self.trees_ = ['']*(self.n_trees)
        # weights for each tree
        self.tree_weight_ = np.zeros(self.n_trees)
        # number of features
        self.number_features = 0

    def fit(self, x, y, feature_names=None):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
            OPTIONAL:
                - feature_names: np array of names
        OUTPUT:
            NONE
        '''

        self.number_features = x.shape[1]
        # inital integer weights as 1
        s_weight = np.ones(x.shape[0])
        # initialize feature names if they are not present
        if feature_names is None or len(feature_names) != self.number_features:
            self.feature_names = np.arange(self.number_features)
        else:
            self.feature_names = feature_names

        # Start boosting
        for i in xrange(self.n_trees):
            tree, s_weight, tree_weight = self._boost(x, y, s_weight)
            # Append estimator error to list
            self.trees_[i] = tree
            # Sample weight error to list
            self.tree_weight_[i] = tree_weight

    def _boost(self, x, y, s_weight):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
            - s_weight: 1d np array of error weights
        OUTPUT:
            - tree: DecisionTreeClassifier
            - s_weight: 1d np array of updated error weights
            - tree_weight: float value that weights the tree
        '''

        # generate a larger x and y data-set, xw and xy, based on the s_weight
        s_weight_min = np.min(s_weight)
        s_weighter = (np.round(s_weight / s_weight_min)).astype(int)
        xw = np.array([[None for _ in xrange(x.shape[1])]
                      for _ in xrange(np.sum(s_weighter))])
        yw = np.array([None for _ in xrange(np.sum(s_weighter))])
        loc = 0
        for i, ww in enumerate(s_weighter):
            if ww != 0:
                xw[loc:loc+ww, :] = x[i, :]
                yw[loc:loc+ww] = y[i]
                loc += ww
        # new tree
        tree = DT(max_depth=1)
        # fit tree
        tree.fit(xw, yw)
        # error in tree
        error = np.array(tree.predict(x) != y)
        # fraction of error in tree
        error_frac = (np.sum((s_weighter).astype(float) * error) /
                    float(np.sum(s_weighter)))
        # computed tree weight
        tree_weight = (self.learn_rate * (np.log((1. - error_frac) /
                    error_frac)))
        # update sample weight
        s_weight *= np.exp(tree_weight * error)

        return tree, s_weight, tree_weight

    def predict(self, x):
        '''
        INPUT:
            - x: 2d np array of features
        OUTPUT:
            - labels: 1d np array of binary labels
        '''
        # predict every tree
        preds = np.array([tree.predict(x)*1. for tree in self.trees_])
        # adjust predictions for fast weighting
        preds[preds == 0] = -1
        # aggregate weights
        labels = np.dot(preds.T, self.tree_weight_) >= 0.0
        return labels
