__author__ = 'igor'

import copy

import numpy as np

from elm.elm import GenELMClassifier
from elm.random_layer import MLPRandomLayer


class ArTwo(object):
    def __init__(self, elm_act_fun, proj_act_fun, depth=7,
                 alpha=0.05, hidden=None):
        self.depth = depth
        self.artwo = []
        self.projection = []
        self.alpha = alpha
        self.elm_act_fun = elm_act_fun
        self.proj_act_fun = proj_act_fun
        self.M = 0
        self.D = 0
        self.C = 0
        if hidden is None:
            self.hidden = self.M
        else:
            self.hidden = hidden

    def fit(self, X, y):
        score = []
        # define activation functions
        # TODO move them somewhere (as a dictionary?)
        linear = (lambda x: x)
        relu = (lambda x: np.maximum(0, x))
        leakyrelu = (lambda x: np.maximum(x, 0.1 * x))
        softplus = (lambda x: np.log(1.0 + np.exp(x)))
        sigmoid = (lambda x: 1.0 / (1.0 + np.exp(-x)))
        tanh = (lambda x: np.tanh(x))
        # scale and save the original X and y
        # is it really correct? how about prediction time?
        # X = StandardScaler().fit_transform(X)
        # save original
        # TODO save as a self variable?
        X_orig = copy.copy(X)
        y_orig = copy.copy(y)
        self.M, self.D = X.shape
        if len(y.shape) == 1:
            self.C = 1
        else:
            self.C = y.shape[1]

        modification = np.zeros((self.C, self.D))
        for k in range(self.depth):
            # print "ArTwo.fit() layer", k
            # get the random layer and k-th level classifier
            random_layer = MLPRandomLayer(n_hidden=self.hidden,
                                          activation_func=self.elm_act_fun)
            self.artwo.append(GenELMClassifier(hidden_layer=random_layer))
            # fit and compute the current train classification
            clf = self.artwo[-1]
            clf.fit(X, y)
            score.append(clf.score(X, y))
            # generate a random projection and modify
            # TODO move all random projections generation to fit() beginning
            self.projection.append(np.random.normal(0, 1, self.C * self.D).reshape((self.C, self.D)))
            out = clf.decision_function(X).reshape((self.M, self.C))
            modification = modification + np.mat(out) * np.mat(self.projection[-1])
            # apply projection
            X = self.proj_act_fun(X_orig + self.alpha * modification)
            # print "ArTwo().fit(): computed layer", k, clf.score(X, y)

        return score

    def predict(self, X):
        X_orig = copy.copy(X)
        modification = np.zeros((self.C, self.D))
        N = X.shape[0]
        for k in range(self.depth):
            clf = self.artwo[k]
            out = clf.decision_function(X).reshape((N, self.C))
            modification = modification + np.mat(out) * np.mat(self.projection[k])
            X = self.proj_act_fun(X_orig + self.alpha * modification)

        return self.artwo[-1].predict(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))

    def decision_function(self, X):
        return self.predict(X)

