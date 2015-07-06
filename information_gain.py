from collections import Counter
import numpy as np


def info_gain(y, y1, y2, metric):

    '''
    INPUT:
        - y: 1d np array of targets
        - y1: 1d np array of targets split
        - y2: 1d np array of targets split
        - metric: function of information gain
    OUTPUT:
        - information gained from split
    '''

    l1 = len(y1) / len(y)
    l2 = len(y2) / len(y)

    return metric(y)-l1*metric(y1)-l2*metric(y2)


def gini(y):
    '''
    INPUT:
        - y: 1d np array of targets
    OUTPUT:
        - measure of how often random selection would
        result in incorrect labels
    '''

    yc = Counter(y)
    p = np.array(yc.values()).astype(float)/len(y)

    return 1.-np.sum(p*p)


def entropy(y):
    '''
    INPUT:
        - y: 1d np array of targets
    OUTPUT:
        - measure of how much disorder is in set
    '''

    yc = Counter(y)
    p = np.array(yc.values()).astype(float)/len(y)

    return -1.*np.sum(p*np.log(p))


def variance(y):
    '''
    INPUT:
        - y: 1d np array of targets
    OUTPUT:
        - measure of how much disorder is in set
    '''

    return np.var(y)