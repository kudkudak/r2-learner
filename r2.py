import numpy as np
import scipy
import sklearn

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator

from functools import partial
from elm import ELM


class R2Learner(BaseEstimator):

    def __init__(self, C=1, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev=False, base_cls=None, is_base_multiclass=False):
        self.name = 'r2svm'
        self.use_prev = use_prev
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
                np.vstack([-self.models_[i].decision_function(X).reshape(1,-1), self.models_[i].decision_function(X).reshape(1,-1)]).T)

            if self.recurrent:
                self._delta += np.dot(self._o[i], self.W[i])
            else:
                self._delta = np.dot(self._o[i], self.W[i])

            if self.use_prev:
                self._X_moved.append(X + self.beta*self._delta)
                X = getattr(self, "_" + self.activation)(self._X_moved[-1])
            else:
                self._X_moved.append(self._X_tr[0] + self.beta*self._delta)
                X = getattr(self, "_" + self.activation)(self._X_moved[-1])

            if self.scale:
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
            # print("WARNING: seeding whole numpy (forced by bug in SVC)")
        self.random_state = np.random.RandomState(self.seed)

        # Models and scalers
        if type(X) == np.ndarray:
            self.scalers_ = [MinMaxScaler((-1.2, 1.2)) for _ in xrange(self.depth)]
        elif type(X) == scipy.sparse.csr.csr_matrix:
            self.scalers_ = [Normalizer(norm='l2') for _ in xrange(self.depth)]
        else:
            raise "Wrong data type, got:", type(X)

        if self.K <= 2:
            self.models_ = [self.base_cls() for _ in xrange(self.depth)]
        #for m in self.models_:
        #    m.set_params(random_state=self.random_state)    # is this necessary?
        # else :
        else:
             if self.is_base_multiclass:
                 self.models_ = [self.base_cls().set_params(random_state=self.random_state) for _ in xrange(self.depth)]
             else:
                 self.models_ = [OneVsRestClassifier(self.base_cls().set_params(random_state=self.random_state), \
                                            n_jobs=1) for _ in xrange(self.depth)]
        
        self.W = W if W else [self.random_state.normal(size=(self.K, X.shape[1])) for _ in range(self.depth - 1)]

        # Prepare data
        if self.scale:
            X = self.scalers_[0].fit_transform(X)
        self._fitted = False

        # Fit
        for i in xrange(self.depth):
            X = self._feed_forward(X, i, Y)

        return self

    def predict(self, X):
        # Prepare data
        if self.scale:
            X = self.scalers_[0].transform(X)

        # Predict
        for i in xrange(self.depth):
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


class R2ELMLearner(R2Learner):
    def __init__(self, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev=False, max_h=100, h=10,
                 fit_h=None):
        self.h = h
        self.max_h = max_h

        if fit_h == None:
            base_cls = partial(ELM, h=self.h, activation='linear')
        else:
            raise NotImplementedError()


        R2Learner.__init__(self, activation=activation, recurrent=recurrent, depth=depth,\
                 seed=seed, beta=beta, scale=scale, use_prev=use_prev, base_cls=base_cls, is_base_multiclass=True)

class R2SVMLearner(R2Learner):
    def __init__(self, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev=False, fit_c=None, C=1, use_linear_svc=False):

		if not use_linear_svc:
		    if fit_c == None:
		        base_cls = partial(SVC, class_weight='auto', kernel='linear', C=C)
		    else:
		        raise NotImplementedError()


		    R2Learner.__init__(self, activation=activation, recurrent=recurrent, depth=depth,\
		             seed=seed, beta=beta, scale=scale, use_prev=use_prev, base_cls=base_cls, is_base_multiclass=False)
		else:
		    if fit_c == None:
		        base_cls = partial(LinearSVC, loss='l1', C=C, class_weight='auto')
		    else:
		        raise NotImplementedError()


		    R2Learner.__init__(self, activation=activation, recurrent=recurrent, depth=depth,\
		             seed=seed, beta=beta, scale=scale, use_prev=use_prev, base_cls=base_cls, is_base_multiclass=True)			
