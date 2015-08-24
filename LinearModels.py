import numpy as np


class LinearModels(object):
    '''
    Model for fitting
    '''
    def __init__(self, hypo='linear', lamda=0, intercept=True):
        '''
        define what type of linear model will be used
        INPUT:
            - hypo: string for model type
            - lamda: float for L2 regularization (Ridge)
            - intercept: bool for fitting an intercept
        '''
        if hypo == 'logistic':
            # hypothesis function
            self.hypo = self.log_hypothesis
            # cost function
            self.cost = self.log_cost
            # cost function gradient
            self.gradient = self.log_cost_gradient

        elif hypo == 'linear':
            # hypothesis function
            self.hypo = self.linear_hypothesis
            # cost function
            self.cost = self.linear_cost
            # cost function gradient
            self.gradient = self.linear_cost_gradient
        # L2 regularization parameter
        self.lamda = lamda
        # bool for fitting an intercept
        self.intercept = intercept
        # number of features
        self.number_features = None
        # coefficient values
        self.coeffs = None

    ###############################################################

    def fit(self, x, y):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
        OUTPUT:
            NONE
        '''
        if self.intercept:
            x = self.add_intercept(x)
        self.number_features = x.shape[1]
        self.coeffs = np.zeros(self.number_features)
        self.gd_opt(x, y)

    def predict(self, x):
        '''
        INPUT:
            - x: 2d np array of features
        OUTPUT:
            - y: 1d np array of targets
        '''
        if self.intercept:
            x = self.add_intercept(x)
        return self.hypo(x, self.coeffs)

    ###############################################################
    '''
    Linear Regression functions taking
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
            - betas: 1d np array of coeffs
    '''

    def linear_hypothesis(self, x, betas):
        '''
        OUTPUT:
            -1d np array of targets
        '''
        return x.dot(betas)

    def linear_cost(self, betas, x, y):
        '''
        OUTPUT:
            -float of cost
        '''
        m = len(y)
        h = self.linear_hypothesis(x, betas)
        r = (h-y)
        cost = np.sum(r*r) / 2. * m
        cost += self.lamda * np.sum(betas ** 2.)
        return cost

    def linear_cost_gradient(self, betas, x, y):
        '''
        OUTPUT:
            -float of gradient
        '''
        m = len(y)
        h = self.linear_hypothesis(x, betas)
        r = (h-y)
        gradient = -(1. / m) * np.dot(x.T, r)
        gradient -= 2. * self.lamda * betas
        return gradient

    ###############################################################
    '''
    Logistical Regression functions taking
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
            - betas: 1d np array of coeffs
    '''

    def log_hypothesis(self, x, betas):
        '''
        OUTPUT:
            - 1d np array of targets
        '''
        return 1.0 / (1.0 + np.exp(-1. * np.dot(x, betas)))

    def log_cost(self, betas, x, y):
        '''
        OUTPUT:
            - float of cost
        '''
        h = self.log_hypothesis(x, betas)
        cost = 1.*(y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
        cost -= self.lamda * np.sum(betas ** 2.)
        return cost

    def log_cost_gradient(self, betas, x, y):
        '''
        OUTPUT:
            - float of gradient
        '''
        h = self.log_hypothesis(x, betas)
        gradient = np.dot(x.T, (y - h))
        gradient -= 2. * self.lamda * betas
        return gradient

    ###############################################################

    def add_intercept(self, x):
        '''
        INPUT:
            - x: 2d np array of features
        OUTPUT:
            - 2d np array + 1 of features and intercept
        '''        
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    ###############################################################


    def sgd_opt(self, x, y):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
        - - Will set optimized self.coeff parameters
        '''
        rounds = 150
        conver = .000001
        alpha = 10.0
        obs = x.shape[0]
        idx = np.arange(obs)
        cost0 = 1e10
        cost1 = 0        
        count = 0
        # stop after more than count cycles of the data
        while count < rounds:
            np.random.shuffle(idx)
            x = x.take(idx, axis=0)
            y = y.take(idx)
            # compute gradient with respect to each observation
            for i in xrange(obs):
                x1 = np.atleast_1d([x[i, :]])
                y1 = np.atleast_1d([y[i]])
                self.coeffs += alpha / obs * self.gradient(self.coeffs, x1, y1)
            cost1 += self.cost(self.coeffs, x, y)
            diff = np.abs((cost1 - cost0) / cost0)
            cost0 = cost1
            # stop if the change in gradient is smaller than threshold
            if diff < conver:
                return
            count += 1
        return
    
    def gd_opt(self, x, y):
        '''
        INPUT:
            - x: 2d np array of features
            - y: 1d np array of targets
        - - Will set optimized self.coeff parameters
        '''
        rounds = 15000
        conver = .000001
        alpha = 0.01
        obs = x.shape[0]
        idx = np.arange(obs)
        cost0 = 1e10
        cost1 = 0        
        count = 0
        
        # stop after more than count cycles of the data
        while count < rounds:
            # compute gradient with respect to each observation
            self.coeffs += alpha * self.gradient(self.coeffs, x, y)
            cost1 = self.cost(self.coeffs, x, y)
            diff = np.abs((cost1 - cost0) / cost0)
            cost0 = cost1
            print self.coeffs
            # stop if the change in gradient is smaller than threshold
            if diff < conver:
                return
            count += 1
        return    