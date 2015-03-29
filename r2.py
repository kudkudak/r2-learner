import numpy as np
import scipy

import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from functools import partial
from elm import ELM

from sklearn.base import BaseEstimator, clone


class MyLinModel(BaseEstimator):
    def __init__(self,w,b):
        assert isinstance(b, (int, long, float)) or len(b.shape) == 1
        self.w=w
        self.b=b

    def fit(self, X, Y):
        pass

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)

    def decision_function(self, X):
        return X.dot(self.w.T) + self.b

def make_rand_vector(dims):
    vec = [np.random.normal(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([[x/mag for x in vec]])


def _r2_compress_model(r2):
    """
    If models are LinearMixins <=> have coefs_ and coef0_ it rewrites to dicts (nope, idk why)//those to MyLinModel. It is still functional model :)
    """
    # #
    #MyLinModel(r2.models_[id].coef_, r2.models_[id].intercept_)
    r2._X_tr = []
    r2._X_moved = []
    r2.X_tr = []
    for id, m in enumerate(r2.models_):
        # I know it should be class testing ok?
        if hasattr(r2.models_[id], 'coef_'):
            if hasattr(r2.models_[id], 'intercept_'):
                r2.models_[id] = {"w": r2.models_[id].coef_, "b":  r2.models_[id].intercept_, "params": r2.models_[id].get_params()}
            elif hasattr(r2.models_[id], 'coef0'):
                r2.models_[id] = {"w": r2.models_[id].coef_, "b":  r2.models_[id].coef0, "params": r2.models_[id].get_params()}
    return r2

class R2Learner(BaseEstimator):
    def __init__(self, C=1, activation='sigmoid', recurrent=True, depth=7, \
                 seed=None, beta=0.1, scale=False, use_prev=False, fit_c=None, base_cls=None,
				fixed_prediction=False, is_base_multiclass=False, switched=False):
        self.name = 'r2svm'
        self.fixed_prediction = fixed_prediction
        self.use_prev = use_prev
        self.fit_c = fit_c
        self.depth = depth
        self.beta = beta
        self.base_cls = base_cls
        self.seed = seed
        self.scale = scale
        self.recurrent = recurrent
        self.C = C
        self.X_tr = []
        self.layer_predictions_ = []
        self.activation = activation  # Minor hack to pickle functions, we will call it by getattr
        self.is_base_multiclass = is_base_multiclass  # Minor hack to know when to wrap in OneVsRest

        # Used in _feed_forward for keeping state
        self._o = []
        self._delta = []
        self._fitted = False
        self._X_moved = []
        self._X_tr = []
        self._prev_C = None
        self.switched = switched


    def _feed_forward(self, X, i, Y=None):
        # Modifies state (_o, _delta, _fitted, _X_tr, _X_moved)
        # Assumes scaled data passed to it (so you have to scale data)

        if i == 0:
            self._o = []
            self._delta = np.zeros(shape=X.shape)
            self._X_tr = [X]
            self._X_moved = [X]

        if not self._fitted:
            if self.fit_c is None:
                self.models_[i].fit(X, Y)
            elif self.fit_c == 'random_cls' or self.fit_c == 'random_cls_centered':
                if i != self.depth - 1:
                    if self.K <= 2:
                        w = make_rand_vector(X.shape[1])
                        if self.fit_c == 'random_cls':
                            b = np.random.uniform(X.min(), X.max())
                        elif self.fit_c == 'random_cls_centered':
                            p = w.dot(X.T)
                            if np.std(p) != 0:
                                b = np.random.normal((p.max() - p.min())/2, np.std(p))
                            else:
                                b = np.random.normal((p.max() - p.min())/2, 1)

                        self.models_[i] = MyLinModel(w, b)
                    else:
                        w = np.hstack([make_rand_vector(X.shape[1]).T for _ in range(self.K)]).T
                        p = w.dot(X.T)

                        if self.fit_c == 'random_cls':
                            b = np.array([np.random.uniform(X.min(), X.max()) for _ in range(self.K)])
                        elif self.fit_c == 'random_cls_centered':
                            b = []
                            for pi in p:
                                b.append(np.random.normal((pi.max() - pi.min())/2, np.std(pi)))
                            b = np.array(b)

                        self.models_[i] = MyLinModel(w, b)
                else:
                    self.models_[i].fit(X, Y)
            elif self.fit_c == 'random' or self.fit_c == 'random_exhaustive':
                if not self.fixed_prediction or i == self.depth - 1:
                    best_C = None
                    best_score = 0.
                    fit_size = 7 if self.fit_c == 'random_exhaustive' else 4
                    if type(self.models_[i]) == ELM:
                        c = [10**j for j in xrange(0, fit_size)]
                    elif type(self.models_[i]) == LinearSVC or type(self.models_[i] == SVC) :
                        c = np.random.uniform(size=fit_size)
                        c = MinMaxScaler((-2, 10)).fit_transform(c) if self.fit_c == 'random_exhaustive' else MinMaxScaler((-2,8)).fit_transform(c)
                        c = [np.exp(x) for x in c]
                        # Add one and previous
                        c = list(set(c).union([1]).union([self._prev_C])) if self._prev_C else list(set(c).union([1]))

                    for j in xrange(fit_size):
                        model = clone(self.models_[i]).set_params(estimator__C=c[j]) if not self.is_base_multiclass \
                                                                                        and self.K > 2 else \
                            clone(self.models_[i]).set_params(C=c[j])
                        score = sklearn.metrics.accuracy_score(model.fit(X,Y).predict(X), Y)
                        #scores = cross_val_score(model, X, Y, scoring='accuracy', \
                        #                         cv=KFold(X.shape[0], shuffle=True, random_state=self.random_state))
                        #score = scores.mean()
                        if score > best_score:
                            best_score = score
                            best_C = c[j]
                    assert best_C is not None
                    self.models_[i].set_params(estimator__C=best_C) if not self.is_base_multiclass and self.K > 2 \
                        else self.models_[i].set_params(C=best_C)
                    self._prev_C = best_C
                    self.models_[i].fit(X, Y)

        if i != self.depth - 1:

            if not self.fixed_prediction:
                self._o.append(self.models_[i].decision_function(X) if self.K > 2 else \
		                           np.vstack([-self.models_[i].decision_function(X).reshape(1, -1),
		                                      self.models_[i].decision_function(X).reshape(1, -1)]).T)
            elif isinstance(self.fixed_prediction, (int, long, float, complex)):
                self._o.append(np.ones(shape=(X.shape[0], self.K)) * self.fixed_prediction)
            else:
                raise NotImplementedError("self.fixed_prediction is wut?")

            if self.recurrent:
                self._delta = sum(np.dot(self._o[j], self.W[i][j]) for j in range(i+1))
            else:
                self._delta = np.dot(self._o[i], self.W[i])

            if self.use_prev:
                self._X_moved.append(X + self.beta * self._delta)
                X = getattr(self, "_" + self.activation)(self._X_moved[-1])
            else:
                self._X_moved.append(self._X_tr[0] + self.beta * self._delta)
                X = getattr(self, "_" + self.activation)(self._X_moved[-1])

            if self.scale:
                if not self._fitted:
                    X = self.scalers_[i + 1].fit_transform(X)
                else:
                    X = self.scalers_[i + 1].transform(X)

            self._X_tr.append(X)
        else:
            self._fitted = True

        return X

    def fit(self, X, Y, W=None):
        self.K = len(set(Y))  # Class number

        # Seed
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            np.random.seed(self.seed)
            # print("WARNING: seeding whole numpy (forced by bug in SVC)")
        self.random_state = np.random.RandomState(self.seed)

        # Models and scalers
        self.scalers_ = [MinMaxScaler((-1, 1)) for _ in xrange(self.depth)]

        if self.K <= 2:
            self.models_ = [self.base_cls() for _ in xrange(self.depth)]
            # for m in self.models_:
            #    m.set_params(random_state=self.random_state)    # is this necessary?
            # else :
        else:
            if self.is_base_multiclass:
                if self.base_cls.func != LogisticRegression:
                    self.models_ = [self.base_cls().set_params(random_state=self.random_state) for _ in xrange(self.depth)]
                else:
                    self.models_ = [self.base_cls() for _ in xrange(self.depth)]
            else:
                raise NotImplementedError, "None base mutliclass models are deprecated."
                # self.models_ = [OneVsRestClassifier(self.base_cls().set_params(random_state=self.random_state), \
                #                                     n_jobs=1) for _ in xrange(self.depth)]

        if self.switched:
            if self.base_cls.func != ELM:
                raise NotImplementedError, "Only switching from ELM to LinearSVC is supported"
            self.models_[-1] = LinearSVC( loss='l1', C=1, class_weight='auto', ).set_params(random_state=self.random_state)

        if self.recurrent:
            self.W = W if W else [[self.random_state.normal(size=(self.K, X.shape[1])) for _ in range(i+1)] \
                                  for i in range(self.depth - 1)]
        else:
            self.W = W if W else [self.random_state.normal(size=(self.K, X.shape[1])) for _ in range(self.depth - 1)]

        # Prepare data
        if self.scale:
            X = self.scalers_[0].fit_transform(X)
        self._fitted = False

        # Fit
        for i in xrange(self.depth):
            X = self._feed_forward(X, i, Y)

        return self

    def predict(self, X, all_layers=False):
        # Prepare data
        if self.scale:
            X = self.scalers_[0].transform(X)

        _X = [X]
        # Predict
        for i in xrange(self.depth):
            X = self._feed_forward(X, i)
            if all_layers and i != self.depth-1: # Last layer is
                _X.append(X)

        if all_layers:
            return [m.predict(X_tr) for m, X_tr in zip(self.models_, _X)]
        else:
            return self.models_[-1].predict(X)

    @staticmethod
    def _tanh(x):
        return 2. / (1. + np.exp(x)) - 1.

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _rbf(x):
        return np.exp(-np.power((x - np.mean(x, axis=0)), 2))

    @staticmethod
    def _01_rbf(x):
        return np.exp(-(np.power(x,2)/2))


def score_all_depths_r2(model, X, Y):
    """
    @returns depth, score_for_this_depth
    """
    return [sklearn.metrics.accuracy_score(Y_pred, Y) for Y_pred in model.predict(X, all_layers=True)]

class R2ELMLearner(R2Learner):
    def __init__(self, activation='sigmoid', recurrent=True, depth=10, \
                 seed=None, beta=0.1, scale=False, fit_c=None, use_prev=False, max_h=100, h=10,
                 fit_h=None, C=100, fixed_prediction=False, switched=False):
        """
        @param fixed_prediction pass float to fix prediction to this number or pass False to learn model
        """
        self.h = h
        self.max_h = max_h

        if fit_h == None:
            base_cls = partial(ELM, h=self.h, activation='linear', C=C)
        else:
            raise NotImplementedError()

        R2Learner.__init__(self, fixed_prediction=fixed_prediction, activation=activation, recurrent=recurrent, depth=depth, \
                           seed=seed, beta=beta, scale=scale, use_prev=use_prev, base_cls=base_cls,
                           is_base_multiclass=True, fit_c=fit_c, C=C, switched=switched)


class R2SVMLearner(R2Learner):
    def __init__(self, activation='sigmoid', recurrent=True, depth=10, seed=None, beta=0.1, scale=False,
                 fixed_prediction=False, use_prev=False, fit_c=None, C=1, use_linear_svc=True, switched=False):
        """
        @param fixed_prediction pass float to fix prediction to this number or pass False to learn model
        """
        if not use_linear_svc:
            raise NotImplementedError("Deprecated. SVC seems much slower for it has to be wrapped as multiclass")            

            base_cls = partial(SVC, class_weight='auto', kernel='linear', C=C)

            R2Learner.__init__(self, fixed_prediction=fixed_prediction, activation=activation, recurrent=recurrent, depth=depth, \
                               seed=seed, beta=beta, scale=scale, fit_c=fit_c, use_prev=use_prev, base_cls=base_cls,
                               is_base_multiclass=False)
        else:
            base_cls = partial(LinearSVC, loss='l1', C=C, class_weight='auto')

            R2Learner.__init__(self, fixed_prediction=fixed_prediction, activation=activation, recurrent=recurrent, depth=depth, \
                               seed=seed, beta=beta, fit_c=fit_c, scale=scale, use_prev=use_prev, base_cls=base_cls,
                               is_base_multiclass=True, switched=switched)


class R2LRLearner(R2Learner):
    def __init__(self, activation='sigmoid', recurrent=True, depth=10, seed=None, beta=0.1, scale=False, \
                 fixed_prediction=False, use_prev=False, logger=None):

        base_cls =  partial(LogisticRegression, fit_intercept=True)

        R2Learner.__init__(self, fixed_prediction=fixed_prediction, activation=activation, recurrent=recurrent, depth=depth, \
                               seed=seed, beta=beta, scale=scale, use_prev=use_prev, base_cls=base_cls,
                               is_base_multiclass=True)
