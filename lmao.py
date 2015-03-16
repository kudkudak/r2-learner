
import sys
import os
sys.path.append("..")

from sklearn import svm, datasets
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.base import BaseEstimator, clone 
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import os
import math
import numpy as np
import sklearn.metrics
from multiprocessing import Pool
from functools import partial

import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler

from data_api import *

from r2 import *




class LMAO(BaseEstimator):
    """Layered Magical Anarchistic Objectifier model"""
    
    def __init__(self, C=1, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, projectors=[], scale=False, base_cls=None, is_base_multiclass=False):
        self.name = 'r2svm'
        self.depth = depth
        self.base_cls = base_cls
        self.seed = seed
        self.scale = scale
        self.recurrent = recurrent
        self.C = C
        self.X_tr = []
        self.projectors=projectors
        self.layer_predictions_ = []
        self.activation = activation # Minor hack to pickle functions, we will call it by getattr
        self.is_base_multiclass = is_base_multiclass # Minor hack to know when to wrap in OneVsRest

        # Used in _feed_forward for keeping state
        self._fitted = False
        self._X_moved = []
        self._X_tr = []

    def _feed_forward(self, X, i, Y=None):
        # Modifies state (_o, _delta, _fitted, _X_tr, _X_moved)
        # Assumes scaled data passed to it (so you have to scale data)

        if i == 0:
            self._o = []
            self._delta = np.zeros(shape=X.shape)
            self._X_tr = [X]
            self._X_moved = [X]
            
        if i != self.depth - 1:
            X = getattr(self, "_" + self.activation)(X + X.dot(self.projectors[i][0])+self.projectors[i][1])

            if self.scale:
                if not self._fitted:
                    X = self.scalers_[i+1].fit_transform(X)
                else:
                    X = self.scalers_[i+1].transform(X)

            self._X_tr.append(X)
        else:
            self.models_[i].fit(X, Y)
            self._fitted = True

        return X

    def fit(self, X, Y):
        self.K = len(set(Y)) # Class number

        # Seed
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            np.random.seed(self.seed)
            # print("WARNING: seeding whole numpy (forced by bug in SVC)")
        self.random_state = np.random.RandomState(self.seed)

        # Models and scalers
        self.scalers_ = [MinMaxScaler((-1.2, 1.2)) for _ in xrange(self.depth)]
        if self.K <= 2:
            self.models_ = [self.base_cls() for _ in xrange(self.depth)]
            for m in self.models_:
                m.set_params(random_state=self.random_state)    # is this necessary?
        else :
            if self.is_base_multiclass:
                self.models_ = [self.base_cls().set_params(random_state=self.random_state) for _ in xrange(self.depth)]
            else:
                self.models_ = [OneVsRestClassifier(self.base_cls().set_params(random_state=self.random_state), \
                                                n_jobs=1) for _ in xrange(self.depth)]

        # Prepare data
        if self.scale:
            X = self.scalers_[0].fit_transform(np.copy(X))
        self._fitted = False

        # Fit
        for i in xrange(self.depth):
            X = self._feed_forward(X, i, Y)

        return self

    def predict(self, X):
        # Prepare data
        if self.scale:
            X = self.scalers_[0].transform(np.copy(X))

        # Predict
        for i in xrange(self.depth - 1):
            X = self._feed_forward(X, i)

        return self.models_[-1].predict(X)

    @staticmethod
    def _tanh(x):
        return 2./(1.+np.exp(x)) - 1.

    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def _rbf(x):
        return np.exp(-np.power((x-np.mean(x, axis=0)),2))

class LMAO2(BaseEstimator):
    """Layered Magical Anarchistic Objectifier model"""
    
    def __init__(self, C=1, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, projectors=[], beta=0.1, scale=False, base_cls=None, is_base_multiclass=False):
        self.name = 'r2svm'
        self.depth = depth
        self.beta = beta
        self.base_cls = base_cls
        self.seed = seed
        self.scale = scale
        self.recurrent = recurrent
        self.C = C
        self.X_tr = []
        self.layer_predictions_ = []
        self.activation = activation # Minor hack to pickle functions, we will call it by getattr
        self.is_base_multiclass = is_base_multiclass # Minor hack to know when to wrap in OneVsRest

        # Used in _feed_forward for keeping state
        self._fitted = False
        self._X_moved = []
        self._X_tr = []

    def _feed_forward(self, X, i, Y=None):
        # Modifies state (_o, _delta, _fitted, _X_tr, _X_moved)
        # Assumes scaled data passed to it (so you have to scale data)

        if i == 0:
            self._o = []
            self._delta = np.zeros(shape=X.shape)
            self._X_tr = [X]
            self._X_moved = [X]
            
        if i != self.depth - 1:
            self._X_moved.append(self._X_tr[0] + self.beta*self._delta)
            X = getattr(self, "_" + self.activation)(X + X.dot(self.projectors[i][0])+self.projectors[i][1])

            if self.scale:
                if not self._fitted:
                    X = self.scalers_[i+1].fit_transform(X)
                else:
                    X = self.scalers_[i+1].transform(X)

            self._X_tr.append(X)
        else:
            self.models_[i].fit(X, Y)
            self._fitted = True

        return X

    def fit(self, X, Y):
        self.K = len(set(Y)) # Class number

        # Seed
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            np.random.seed(self.seed)
            # print("WARNING: seeding whole numpy (forced by bug in SVC)")
        self.random_state = np.random.RandomState(self.seed)

        # Models and scalers
        self.scalers_ = [MinMaxScaler((-1.2, 1.2)) for _ in xrange(self.depth)]
        if self.K <= 2:
            self.models_ = [self.base_cls() for _ in xrange(self.depth)]
            for m in self.models_:
                m.set_params(random_state=self.random_state)    # is this necessary?
        else :
            if self.is_base_multiclass:
                self.models_ = [self.base_cls().set_params(random_state=self.random_state) for _ in xrange(self.depth)]
            else:
                self.models_ = [OneVsRestClassifier(self.base_cls().set_params(random_state=self.random_state), \
                                                n_jobs=1) for _ in xrange(self.depth)]

        # Prepare data
        if self.scale:
            X = self.scalers_[0].fit_transform(np.copy(X))
        self._fitted = False

        # Fit
        for i in xrange(self.depth):
            X = self._feed_forward(X, i, Y)

        return self

    def predict(self, X):
        # Prepare data
        if self.scale:
            X = self.scalers_[0].transform(np.copy(X))

        # Predict
        for i in xrange(self.depth - 1):
            X = self._feed_forward(X, i)

        return self.models_[-1].predict(X)

    @staticmethod
    def _tanh(x):
        return 2./(1.+np.exp(x)) - 1.

    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def _rbf(x):
        return np.exp(-np.power((x-np.mean(x, axis=0)),2))
