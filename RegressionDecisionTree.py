import numpy as np
from collections import Counter
from information_gain import variance, info_gain
from LinearModels import LinearModels as lm


class RegressionDecisionTree(object):
    '''
    Tree for classifying
    '''
    def __init__(self, conv_thres=0.0, leaf_min=None, max_depth=None,
                 linear_regressor=False):
        '''
        build empty tree
        INPUT:
            - split_metric: string for how to split
            - conv_thres: float for when the information change
                         is small enough to leaf
            - leaf_min: int for smallest number y values to define a leaf
            - max_depth: int for max size of a branch
            - linear_regressor: bool for whether to fit a lienar regressor
                defaults lieaf_min = 2
        '''
        # the start of the tree
        self.root = None
        # names of all the features
        self.feature_names = None
        # whether a feature is a category or number
        self.categorical = None
        # number of features
        self.number_features = None
        # number of observations
        self.observations = None
        # metric for spliting
        self.split_metric = variance
        # threshold for information gain convergence
        self.conv_thres = conv_thres
        # fit linear regressor on leaves
        self.linear_regressor = linear_regressor
        # minimum number of targets in each leaf
        if (self.linear_regressor) & leaf_min is None:
            self.leaf_min = 2
        else:
            self.leaf_min = leaf_min
        # max depth of tree
        self.max_depth = max_depth
        # number of nodes
        self.nodes = 0
        # measure tree depth
        self.measured_depth = 0

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
        # initialize fit data details
        self.number_features = x.shape[1]
        self.observations = x.shape[0]
        # initialize feature names if they are not present
        if feature_names is None or len(feature_names) != self.number_features:
            self.feature_names = np.arrage(self.number_features)
        else:
            self.feature_names = feature_names
        # initialize feature types
        self.categorical = [isinstance(i, str) or
                            isinstance(i, bool) or
                            isinstance(i, unicode)
                            for i in x[0]]
        # build the tree
        self.root = self._build_tree(x, y)
        print 'total nodes = {}'.format(self.nodes-1)
        print 'total depth = {}'.format(self.measured_depth+1)

    def predict(self, x):
        '''
        INPUT:
            - x: 2d np array of features
        OUTPUT:
            - y: 1d np array of targets
        '''
        # recursively predict for each x value
        y_pred = np.apply_along_axis(self.root.node_value, axis=1, arr=x)

        return y_pred

    def _build_tree(self, x, y, depth=0):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
        OUTPUT:
            - Node of Tree
        '''
        # create a tree node
        node = NodeR()
        # find where to best split that node
        idx, val, splt = self._select_split_idx(x, y)

        self.nodes += 1
        # test if a branch is over
        if (len(np.unique(y)) == 1) or (idx is None) \
                or (depth == self.max_depth):
            # if it is set final values
            node.leaf = True
            node.value_array = y
            if (self.linear_regressor) & (len(y) > 2):
                node.linear_regressor = lm(hyp='linear')
                node.linear_regressor.fit(x, y)
            else:
                node.yvalue = np.mean(y)
        else:
            # if it is not set branches recursively
            x1, x2, y1, y2 = splt
            node.col = idx
            node.name = self.feature_names[idx]
            node.val = val
            node.categorical = self.categorical[idx]
            node.left = self._build_tree(x1, y1, depth+1)
            node.right = self._build_tree(x2, y2, depth+1)
        # node with connected nodes
            if depth > self.measured_depth:
                self.measured_depth = depth
        return node

    def _make_split(self, x, y, idx, val):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
            - idx: int feature index to make split
            - val: arbitary type feature value to make split
        OUTPUT:
            - x1: 2d np array of features
            - x2: 2d np array of features
            - y1: 1d np array of targets
            - y2: 1d np array of targets
        '''
        # test if feature is categorical
        # then find the mask that that splits the data
        if self.categorical[idx]:
            cmask = x[:, idx] == val
        else:
            cmask = x[:, idx] < val
        # apply and return the masked arrays
        return x[cmask == True, :], x[cmask == False, :],\
            y[cmask == True], y[cmask == False]

    def _select_split_idx(self, x, y):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
        OUTPUT:
            - idx: int feature index to make split
            - val: arbitary type feature value to make split
            - splt: two 2d np arrays of features,
                    two 1d np arrays of targets
        '''
        # test every feature's value to find optimial place to apply split
        max_gain = -1e10
        # iterate through every feature
        for i in xrange(self.number_features):
            for split_val in np.unique(x[:, i]):
                # test splits
                x1, x2, y1, y2 = self._make_split(x, y, i, split_val)
                # check for leaf size if restictions are present
                if self.leaf_min is not None:
                    if len(y1) < self.leaf_min or len(y2) < self.leaf_min:
                        continue
                gain = info_gain(y, y1, y2, self.split_metric)
                if gain > max_gain:
                    max_gain = gain
                    idx = i
                    val = split_val

        # test if splitting improves information gain
        if max_gain > self.conv_thres:
            return idx, val, self._make_split(x, y, idx, val)
        else:
            return None, None, None


class NodeR(object):
    '''
    A node in a tree
    '''

    def __init__(self):
        # index of feature to split on
        self.col = None
        # value of self.col to split on
        self.val = None
        # (bool) whether or not a node is a categorical feature
        self.categorical = True
        # a node split left
        self.left = None
        # a node split right
        self.right = None
        # whether there are no left or right splits
        self.leaf = False
        # y value of leaf node
        self.yvalue = None
        # count of all y values
        self.value_array = None
        # linear regressor
        node.linear_regressor = None

    def node_value(self, x):
        '''
        INPUT
            -
        '''
        if self.leaf:
            # final tree value
            if node.linear_regressor is None:
                return self.yvalue
            else:
                return node.linear_regressor.predict(x)
        else:
            # x values with which to decide splits
            col_val = x[self.col]

        # default to left node being a 'equal' response path
        if self.categorical:
            if col_val == self.val:
                return self.left.node_value(x)
            else:
                return self.right.node_value(x)
        # default to left node being a 'lt' response path
        else:
            if col_val < self.val:
                return self.left.node_value(x)
            else:
                return self.right.node_value(x)







