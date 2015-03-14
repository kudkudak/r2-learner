import numpy as np

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, clone

from functools import partial
from elm import ELM

class R2SVMLearner(BaseEstimator):

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


    def fit(self, X, Y):
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

        # Prepare data
        X_mod = X
        o = []
        delta = np.zeros(shape=X.shape)
        self.W = []
        self.X_tr = [X_mod]
        self.X_moved = [X_mod]

        # Fit
        for i in xrange(self.depth):
            X_mod = self.scalers_[i].fit_transform(X_mod) if self.scale else X_mod

            if self.fit_c == 'grid':
                if self.K > 2 :
                    grid = GridSearchCV(self.models_[i], {'estimator__C': [np.exp(d) for d in xrange(-2,6)]}, \
                                        cv=KFold(X_mod.shape[0], n_folds=3, shuffle=True, random_state=self.random_state), n_jobs=1)
                else :
                    grid = GridSearchCV(self.models_[i], {'C': [np.exp(d) for d in xrange(-2,6)]}, \
                                        cv=KFold(X_mod.shape[0], n_folds=3, shuffle=True, random_state=self.random_state), n_jobs=1)
                grid.fit(X_mod,Y)
                self.models_[i] = grid
            elif self.fit_c == 'random':
                best_C = None
                best_score = 0.
                c = np.random.uniform(size=10)
                c = MinMaxScaler((-2, 10)).fit_transform(c)
                c = [ np.exp(x) for x in c ]
                for j in xrange(10) :
                    model = clone(self.models_[i]).set_params(estimator__C=c[j]) if self.K > 2 else clone(self.models_[i]).set_params(C=c[j])
                    scores = cross_val_score(model, X_mod, Y, scoring='accuracy', cv=KFold(X_mod.shape[0], shuffle=True, random_state=self.random_state))
                    score = scores.mean()
                    if score > best_score :
                        best_score = score
                        best_C = c[j]
                assert best_C is not None
                self.models_[i].set_params(estimator__C=best_C) if self.K > 2 else self.models_[i].set_params(C=best_C)
                self.models_[i].fit(X_mod, Y)
            else:
                self.models_[i].fit(X_mod, Y)

            if self.depth != -1:
                o.append(self.models_[i].decision_function(X_mod) if self.K > 2 else \
                    np.hstack([-self.models_[i].decision_function(X_mod), self.models_[i].decision_function(X_mod)]))

                self.W.append(self.random_state.normal(size=(self.K, X.shape[1])))

                if self.recurrent:
                    delta += np.dot(o[i], self.W[i])
                else:
                    delta = np.dot(o[i], self.W[i])

                if self.use_prev:
                    X_mod = getattr(self, "_" + self.activation)(X_mod + self.beta*delta)
                else:
                    self.X_moved.append((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)
                    X_mod = getattr(self, "_" + self.activation)(self.X_moved[-1])
                self.X_tr.append(X_mod)

        return self

    def predict(self, X, all_layers=False):
        # Prepare data
        X_mod = X
        o = []
        delta = np.zeros(shape=X.shape)

        self.X_tr = [X_mod]
        self.X_moved = [X_mod]

        for i in xrange(self.depth-1):
            X_mod = self.scalers_[i].transform(X_mod) if self.scale else X_mod

            oi = self.models_[i].decision_function(X_mod)
            o.append(oi if self.K > 2 else \
                np.hstack([-oi, oi]))

            if all_layers :
                self.layer_predictions_.append(self.models_[i].predict(X_mod))

            if self.recurrent:
                delta += np.dot(o[i], self.W[i])
            else:
                delta = np.dot(o[i], self.W[i])

            if not self.use_prev:
                self.X_moved.append((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)
                X_mod = getattr(self, "_" + self.activation)(self.X_moved[-1])
            else:
                X_mod = getattr(self, "_" + self.activation)(X_mod + self.beta*delta)

            self.X_tr.append(X_mod)

        X_mod = self.scalers_[self.depth-1].transform(X_mod) if self.scale else X_mod

        self.layer_predictions_.append(self.models_[-1].predict(X_mod))
        return self.models_[-1].predict(X_mod)

    @staticmethod
    def _tanh(x):
        return 2./(1.+np.exp(x)) - 1.

    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def _rbf(x):
        return np.exp(-np.power((x-np.mean(x, axis=0)),2))



class R2ELMLearner(BaseEstimator):
    def __init__(self, h=60, activation='rbf', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev = False, max_h=100,
                 fit_h=None):
        self.name = 'r2elm'
        self.fit_h = fit_h
        self.use_prev = use_prev
        self.depth = depth
        self.beta = beta
        self.seed = seed
        self.scale = scale
        self.recurrent = recurrent
        self.h = h
        self.X_tr = []
        self.layer_predictions_ = []
        self.max_h = max_h

        # Seed
        if self.seed is None:
            self.seed = np.random.randint(0, np.iinfo(np.int32).max)

        self.random_state = np.random.RandomState(self.seed)

        self.base_cls = partial(ELM, h=self.h, activation='linear', seed=self.seed)

        self.activation = activation


    def fit(self, X, Y):
        self.K = len(set(Y)) # Class number

        # Models and scalers
        self.scalers_ = [MinMaxScaler((-1,1)) for _ in xrange(self.depth)]
        self.models_ = [self.base_cls() for _ in xrange(self.depth)]

        # Prepare data
        X_mod = X
        o = []
        delta = np.zeros(shape=X.shape)
        self.W = []
        self.X_tr = [X_mod]

        # Fit
        for i in xrange(self.depth):
            X_mod = self.scalers_[i].fit_transform(X_mod) if self.scale else X_mod
            if self.fit_h == 'grid':
                grid = GridSearchCV(self.models_[i], {'h': [h for h in xrange(10, self.max_h+1, 10)]}, \
                                    cv=KFold(X_mod.shape[0], n_folds=3, shuffle=True, random_state=self.random_state), n_jobs=1)
                grid.fit(X_mod,Y)
                self.models_[i] = grid
            elif self.fit_h == 'random':
                best_h = None
                best_score = 0.
                h = np.random.uniform(size=10)
                h = MinMaxScaler((10, self.max_h)).fit_transform(h)
                for j in xrange(10) :
                    model = clone(self.models_[i]).set_params(h=h[j])
                    scores = cross_val_score(model, X_mod, Y, scoring='accuracy', cv=KFold(X_mod.shape[0], shuffle=True, random_state=self.random_state))
                    score = scores.mean()
                    if score > best_score :
                        best_score = score
                        best_h = h[j]
                assert best_h is not None
                self.models_[i].set_params(h=best_h)
                self.models_[i].fit(X_mod, Y)
            else:
                self.models_[i].fit(X_mod, Y)

            if i != self.depth -1 :
                o.append(self.models_[i].decision_function(X_mod) if self.K > 2 else \
                    np.hstack([-self.models_[i].decision_function(X_mod), self.models_[i].decision_function(X_mod)]))

                self.W.append(self.random_state.normal(size=(self.K, X.shape[1])))

                if self.recurrent:
                    delta += np.dot(o[i], self.W[i])
                else:
                    delta = np.dot(o[i], self.W[i])

                if self.use_prev:
                    X_mod = getattr(self, "_"+self.activation)(X_mod + self.beta*delta)
                else:
                    X_mod = getattr(self, "_"+self.activation)((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)

                self.X_tr.append(X_mod)

        return self

    def predict(self, X, all_layers=False):
        # Prepare data
        X_mod = X
        o = []
        delta = np.zeros(shape=X.shape)

        for i in xrange(self.depth-1):
            X_mod = self.scalers_[i].transform(X_mod) if self.scale else X_mod

            oi = self.models_[i].decision_function(X_mod)
            o.append(oi if self.K > 2 else \
                np.hstack([-oi, oi]))

            if all_layers :
                self.layer_predictions_.append(self.models_[i].predict(X_mod))

            if self.recurrent:
                delta += np.dot(o[i], self.W[i])
            else:
                delta = np.dot(o[i], self.W[i])

            if not self.use_prev:
                X_mod = getattr(self, "_"+self.activation)((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)
            else:
                X_mod = getattr(self, "_"+self.activation)(X_mod + self.beta*delta)

        X_mod = self.scalers_[self.depth-1].transform(X_mod) if self.scale else X_mod

        self.layer_predictions_.append(self.models_[-1].predict(X_mod))
        return self.models_[-1].predict(X_mod)

    @staticmethod
    def _tanh(x):
        return 2./(1.+np.exp(x)) - 1.

    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def _rbf(x):
        return np.exp(-np.power((x-np.mean(x, axis=0)),2))