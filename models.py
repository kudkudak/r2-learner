import sklearn
from functools import partial
from sklearn.svm import SVC
import numpy as np
from scipy import linalg as la
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator


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

class LinELM:

    def __init__(self, h=60):
        self.h = h

    def fit(self, X, y):
        self.b = LabelBinarizer()
        self.W = np.random.normal(size=X.shape[1] * self.h).reshape(X.shape[1], self.h)
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


def _tanh(x) :
    return 2./(1.+np.exp(x)) - 1.

def _sigmoid(x) :
    return 1.0/(1.0 + np.exp(-x))

#TODO: fit C
class R2SVMLearner(BaseEstimator):
    def __init__(self, kernel='linear', C=1, activation='sigmoid', recurrent=True, depth=7,\
                 seed=None, beta=0.1, scale=False, use_prev = False, jobs=1):

        self.use_prev = use_prev
        self.depth = depth
        self.beta = beta
        self.base_cls = partial(SVC, class_weight='auto', kernel='linear', C=C)
        self.seed = seed
        self.scale = scale
        self.activation = activation
        self.recurrent = recurrent
        self.C = C
        self.jobs = jobs
        self.X_tr = []
        self.layer_coefs_ = []
        self.fit_layer_scores_ = []
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
                m.set_params(random_state=self.random_state)
        else :
            self.models_ = [OneVsRestClassifier(self.base_cls().set_params(random_state=self.random_state), n_jobs=self.jobs) for _ in xrange(self.depth)]

        # Prepare data
        X_mod = X
        o = []
        delta = np.zeros(shape=X.shape)
        self.W = []
        self.X_tr = [X_mod]

        # Fit
        for i in xrange(self.depth):
            X_mod = self.scalers_[i].fit_transform(X_mod) if self.scale else X_mod
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

    def predict(self, X):
        # Prepare data
        X_mod = X
        o = []
        delta = np.zeros(shape=X.shape)

        all_layers=False

        for i in xrange(self.depth-1):
            X_mod = self.scalers_[i].transform(X_mod) if self.scale else X_mod

            oi = self.models_[i].decision_function(X_mod)
            o.append(oi if self.K > 2 else \
                np.hstack([-oi, oi]))

            if all_layers :
                self.layer_predictions_.append(self.models_[i].predict(X))

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

    def fit_predict(self, X, Y, score_func=sklearn.metrics.accuracy_score):
        self.K = len(set(Y)) # Class number

        # Seed
        if self.seed is None: self.seed = np.random.randint(0, np.iinfo(np.int32).max)
        self.random_state = np.random.RandomState(self.seed)

        # Models and scalers
        self.scalers_ = [MinMaxScaler() for _ in xrange(self.depth)]
        if self.K <= 2:
            self.models_ = [self.base_cls() for _ in xrange(self.depth)]
            for m in self.models_:
                m.set_params(random_state=self.random_state)
        else :
            self.models_ = [OneVsRestClassifier(self.base_cls().set_params(random_state=self.random_state), \
                                                n_jobs=self.jobs) for _ in xrange(self.depth)]

        # Prepare data
        X_mod = X
        o = []
        delta = np.zeros(shape=X.shape)
        self.W = []
        self.X_tr.append(X_mod)
        self.layer_coefs_ = []
        self.fit_layer_scores_ = []

        # Fit
        for i in xrange(self.depth):
            X_mod = self.scalers_[i].fit_transform(X_mod) if self.scale else X_mod
            self.models_[i].fit(X_mod, Y)

            self.layer_scores_.append(score_func(Y, self.models_[i].predict(X_mod)))
            self.layer_coefs_.append(self.models_[i].coef_)

            o.append(self.models_[i].decision_function(X_mod) if self.K > 2 else \
                np.hstack([-self.models_[i].decision_function(X_mod), self.models_[i].decision_function(X_mod)]))

            self.W.append(self.random_state.normal(size=(self.K, X.shape[1])))

            if self.recurrent:
                delta += np.dot(o[i], self.W[i])
            else:
                raise NotImplementedError()

            if self.use_prev:
                X_mod = self.activation(X_mod + self.beta*delta)
            else:
                X_mod = self.activation((self.scalers_[0].transform(X) if self.scale else X) + self.beta*delta)

            self.X_tr.append(X_mod)

        return self