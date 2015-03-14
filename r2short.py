import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator

from functools import partial

class R2SVMLearnerShort(BaseEstimator):

    def __init__(self, C=1, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev = False, fit_c=None):
        self.name = 'r2svm'
        self.fit_c = fit_c
        self.use_prev = use_prev
        self.depth = depth
        self.beta = beta
        self.base_cls = partial(SVC, class_weight='auto', kernel='linear', C=C)
        self.seed = seed
        self.scale = scale
        self.recurrent = recurrent
        self.C = C
        self.X_tr = []
        self.layer_predictions_ = []
        self.activation = activation # Minor hack to pickle functions, we will call it by getattr

        # Used in _feed_forward for keeping state
        self._o = []
        self._delta = []
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


        if not self._fitted:
            self.models_[i].fit(X, Y)

        if i != self.depth - 1:

            self._o.append(self.models_[i].decision_function(X) if self.K > 2 else \
                np.hstack([-self.models_[i].decision_function(X), self.models_[i].decision_function(X)]))

            if self.recurrent:
                self._delta += np.dot(self._o[i], self.W[i])
            else:
                self._delta = np.dot(self._o[i], self.W[i])

            if self.use_prev:
                self._X_moved.append(X + self.beta*self._delta)
                X = getattr(self, "_" + self.activation)(self._X_moved[-1])
            else:
                # TODO: fix performance
                self._X_moved.append((self.scalers_[0].transform(X) if self.scale else X) + self.beta*self._delta)
                X = getattr(self, "_" + self.activation)(self._X_moved[-1])

            if not self._fitted:
                X = self.scalers_[i+1].fit_transform(X)
            else:
                X = self.scalers_[i+1].transform(X)

            self._X_tr.append(X)
        else:
            self._fitted = True

        return X

    def fit(self, X, Y, W=None):
        self.K = len(set(Y)) # Class number

        # Seed
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            np.random.seed(self.seed)
            print("WARNING: seeding whole numpy (forced by bug in SVC)")
        self.random_state = np.random.RandomState(self.seed)

        # Models and scalers
        self.scalers_ = [MinMaxScaler((-1,1)) for _ in xrange(self.depth)]
        if self.K <= 2:
            self.models_ = [self.base_cls() for _ in xrange(self.depth)]
            for m in self.models_:
                m.set_params(random_state=self.random_state)    # is this necessary?
        else :
            self.models_ = [OneVsRestClassifier(self.base_cls().set_params(random_state=self.random_state), \
                                                n_jobs=1) for _ in xrange(self.depth)]
        self.W = W if W else [self.random_state.normal(size=(self.K, X.shape[1])) for _ in range(self.depth - 1)]

        # Prepare data
        X = self.scalers_[0].fit_transform(np.copy(X))
        self._fitted = False

        # Fit
        for i in xrange(self.depth):
            X = self._feed_forward(X, i, Y)

        self.X_moved = self._X_moved
        self.X_tr = self._X_tr

        return self



    def predict(self, X):
        # Prepare data
        X = self.scalers_[0].transform(np.copy(X))

        # Predict
        for i in xrange(self.depth - 1):
            X = self._feed_forward(X, i)

        self.X_moved = self._X_moved
        self.X_tr = self._X_tr

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


