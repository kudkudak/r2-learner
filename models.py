import sklearn
import numpy as np

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, clone

from functools import partial
from scipy import linalg as la
from matplotlib import pyplot as plt


def _tanh(x):                      # these are needed for multiprocessing purposes
    return 2./(1.+np.exp(x)) - 1.

def _sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class R2SVMLearner(BaseEstimator):
    def __init__(self, C=1, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev = False, fit_c=None):

        self.fit_c = None
        self.use_prev = use_prev
        self.depth = depth
        self.beta = beta
        self.base_cls = partial(SVC, class_weight='auto', kernel='linear', C=C)
        self.seed = seed
        self.scale = scale
        self.activation = activation
        self.recurrent = recurrent
        self.C = C
        self.X_tr = []
        self.layer_predictions_ = []

        if activation == 'tanh':
            # self.activation = lambda x: 2./(1.+np.exp(x)) - 1.
            self.activation = _tanh
        elif activation == 'sigmoid':
            # self.activation = lambda x: 1.0/(1.0 + np.exp(-x))
            self.activation = _sigmoid
        else:
            self.activation = activation

    def fit(self, X, Y):
        self.K = len(set(Y)) # Class number

        # Seed
        if self.seed is None: self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        self.random_state = np.random.RandomState(self.seed)

        # Models and scalers
        self.scalers_ = [MinMaxScaler() for _ in xrange(self.depth)]
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
            elif self.fit_c == 'random' :
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

            o.append(self.models_[i].decision_function(X_mod) if self.K > 2 else \
                np.hstack([-self.models_[i].decision_function(X_mod), self.models_[i].decision_function(X_mod)]))

            self.W.append(self.random_state.normal(size=(self.K, X.shape[1])))

            if self.recurrent:
                delta += np.dot(o[i], self.W[i])
            else:
                delta = np.dot(o[i], self.W[i])

            if self.use_prev:
                X_mod = self.activation(X_mod + self.beta*delta)
            else:
                X_mod = self.activation((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)

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
                X_mod = self.activation((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)
            else:
                X_mod = self.activation(X_mod + self.beta*delta)

        X_mod = self.scalers_[self.depth-1].transform(X_mod) if self.scale else X_mod

        self.layer_predictions_.append(self.models_[-1].predict(X_mod))
        return self.models_[-1].predict(X_mod)


def _elm_vectorized_rbf(X, W, B):
    WS = np.array([np.sum(np.multiply(W,W), axis=0)])
    XS = np.array([np.sum(np.multiply(X,X), axis=1)]).T[0]
    return np.exp(-np.multiply(B, -2*X.dot(W) + WS + XS))

def _elm_sigmoid(X, W, B):
    return _sigmoid(X.dot(W) + B)


class ELM(BaseEstimator):

    def __init__(self, h=60, activation='linear', random_state=None):
        self.h = h
        self.activation = activation
        self.random_state = random_state if random_state is not None \
            else np.random.RandomState(np.random.randint(0, np.iinfo(np.int32).max))

        assert activation in ['rbf', 'sigmoid', 'linear']

    def fit(self, X, y):
        self.b = LabelBinarizer()
        self.W = self.random_state.normal(size=(X.shape[1], self.h))
        self.B = self.random_state.normal(size=self.h)
        if self.activation == 'rbf':
            H = _elm_vectorized_rbf(X, self.W, self.B)
        elif self.activation == 'sigmoid':
            H = _elm_sigmoid(X, self.W, self.B)
        else :
            H = X.dot(self.W)
        self.b.fit(y)
        self.beta = la.pinv(H).dot(self.b.transform(y))

    def decision_function(self, X):
        if self.activation == 'rbf':
            return _elm_vectorized_rbf(X, self.W, self.B).dot(self.beta)
        elif self.activation == 'sigmoid':
            return _elm_sigmoid(X, self.W, self.B).dot(self.beta)
        else :
            return X.dot(self.W).dot(self.beta)

    def predict(self, X):
        return self.b.inverse_transform(self.decision_function(X))


class R2ELMLearner(BaseEstimator):
    def __init__(self, h=60, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev = False, max_h=100,
                 fit_h=None):

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
        if self.seed is None: self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        self.random_state = np.random.RandomState(self.seed)

        self.base_cls = partial(ELM, h=self.h, activation='linear', random_state=self.random_state)

        if activation == 'tanh':
            self.activation = _tanh
        elif activation == 'sigmoid':
            self.activation = _sigmoid
        else:
            self.activation = activation

    def fit(self, X, Y):
        self.K = len(set(Y)) # Class number

        # Models and scalers
        self.scalers_ = [MinMaxScaler() for _ in xrange(self.depth)]
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

            o.append(self.models_[i].decision_function(X_mod) if self.K > 2 else \
                np.hstack([-self.models_[i].decision_function(X_mod), self.models_[i].decision_function(X_mod)]))

            self.W.append(self.random_state.normal(size=(self.K, X.shape[1])))

            if self.recurrent:
                delta += np.dot(o[i], self.W[i])
            else:
                delta = np.dot(o[i], self.W[i])

            if self.use_prev:
                X_mod = self.activation(X_mod + self.beta*delta)
            else:
                X_mod = self.activation((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)

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
                X_mod = self.activation((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)
            else:
                X_mod = self.activation(X_mod + self.beta*delta)

        X_mod = self.scalers_[self.depth-1].transform(X_mod) if self.scale else X_mod

        self.layer_predictions_.append(self.models_[-1].predict(X_mod))
        return self.models_[-1].predict(X_mod)



### ======================================================== OLD ==================================================================

class LinELM(BaseEstimator):

    def __init__(self, h=60, random_state=None):
        self.h = h
        self.random_state = random_state if random_state is not None \
            else np.random.RandomState(np.random.randint(0, np.iinfo(np.int32).max))

    def fit(self, X, y):
        self.b = LabelBinarizer()
        self.W = self.random_state.normal(size=X.shape[1] * self.h).reshape(X.shape[1], self.h)
        H = X.dot(self.W)
        self.b.fit(y)
        self.beta = la.pinv(H).dot(self.b.transform(y))

    def decision_function(self, X):
        return X.dot(self.W).dot(self.beta)

    def predict(self, X):
        return self.b.inverse_transform(self.decision_function(X))


class R2Lin:

    def __init__(self, k, estimator=partial(SVC, kernel='linear', C=1), recurrent=False, scale=False):
        self.k = k
        self.clf = estimator
        self.recurrent = recurrent
        self.activation = lambda x : 1.0 / (1.0 + np.exp(-x))
#        self.activation = lambda x : 1.0/np.sqrt(2*np.pi) * np.exp( - (x)**2  )
        self.alpha = 0.04
        self.scale = scale

    def fit(self, X, y):
        self.Wi = []
        self.clfs = []
        self.scalers = []
        self.C = len(set(y))
        Xi = X
        di = np.zeros(X.shape[0] * X.shape[1]).reshape(X.shape)
        for i in range(self.k):

            if self.C <= 2:
            	clf = self.clf()
            else :
            	clf = OneVsRestClassifier(self.clf())
            scaler = MinMaxScaler()
            if self.scale:
                Xi = scaler.fit_transform(Xi)
            self.scalers.append(scaler)
            clf.fit(Xi, y)
            self.clfs.append(clf)
            Oi = clf.decision_function(Xi)
            if self.C == 2:
                Oi = np.hstack([Oi,-Oi])
            Wi = np.random.normal(size=X.shape[1] * self.C).reshape(self.C, X.shape[1])
            self.Wi.append(Wi)

            if self.recurrent:
                di += Oi.dot(Wi)
            else:
                di = Oi.dot(Wi)
            Xi = self.activation(X + self.alpha * di)
        clf = self.clf()
        scaler = MinMaxScaler()
        if self.scale:
            Xi = scaler.fit_transform(Xi)
        self.scalers.append(scaler)
        clf.fit(Xi, y)
        self.clfs.append(clf)
        self.Wi.append(None)
        return self

    def predict(self, X):
        Xi = X
        di = np.zeros(X.shape[0] * X.shape[1]).reshape(X.shape)
        for clf, Wi, scaler in zip(self.clfs, self.Wi, self.scalers):

            if self.scale:
                Xi = scaler.transform(Xi)

            if Wi is None:
                return clf.predict(Xi)

            Oi = clf.decision_function(Xi)
            if self.C == 2:
                Oi = np.hstack([Oi,-Oi])

            if self.recurrent:
                di += Oi.dot(Wi)
            else:
                di = Oi.dot(Wi)
            Xi = self.activation(X + self.alpha * di)



    def __str__(self):
        return 'R2Lin' + str(self.clf) + ' ' + str(self.k)


class R2SVM():

    def __init__(self,l=2,C=1,beta=0.1):
        self.beta = beta
        self.layers = []
        for i in xrange(l):
            self.layers.append(SVC(C=C,kernel='linear',class_weight='auto'))

    def sigmoid(self,x):
        return 2/(1+np.exp(x)) - 1

    def get_input(self, x, layer, o ):

        if layer == 1: return x
        else:
            if o == None:
                o = self.get_output(x, 1, None)
                for previous in xrange(layer-2):
                    o = np.concatenate((o,self.get_output(x, previous+2,o)),1)
            move = ( np.array(o).dot( np.array(self.projections[layer-2])))
            x_propagated = self.sigmoid(x+self.beta*move)

        return x_propagated

    def get_output(self, x, layer, o = None):
        if layer != 1 and o == None:
            o = self.get_output(x, 1, None)
            for previous in xrange(layer-2):
                o = np.concatenate((o,self.get_output(x, previous+2,o)),1)
        inp = self.get_input(x, layer, o)
        out = np.round( self.layers[layer-1].decision_function(inp),2)
        if (1==out.shape[1]):
            tmp_out = []
            for x in out:
                tmp_out.append([-x[0],x[0]])
            out = tmp_out
        return out

    def fit(self,x,y):
        classes = len(np.unique(y))
        features = len(x[0])
        self.projections = []
        for i in xrange(len(self.layers)-1):
            self.projections.append(np.random.normal(size=(classes*(i+1),features)))
        for i in xrange(len(self.layers)):
            if i==0:
                self.layers[0].fit(x,y)
                o = self.get_output(x, 1, None)
                continue
            self.layers[i].fit( self.get_input(x, i+1, o), y)
            o = np.concatenate((o,self.get_output(x, i+1, o)),1)

        return self

    def classify_internal(self,x,layer):
        return [np.argmax(p) for p in self.get_output( x, layer )]


    def predict(self,x):
        return self.classify_internal( x, len(self.layers) )

    def drawdata(self, x, y):
        _x = np.array(self.get_input(x,len(self.layers),None))
        plt.scatter(_x[:,0].tolist(),_x[:,1].tolist(),c=y)
        plt.show()

    def score(self, X, Y):
        return sklearn.metrics.accuracy_score(Y, self.predict(X))