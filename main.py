
"""
THIS IS A TEST FILE, I KNOW IT'S A MESS
"""

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler
from data_api import *
from models import R2SVMLearner, R2ELMLearner, ELM
import sklearn
from sklearn.grid_search import GridSearchCV
from fit_r2svm import fit_r2svm_on_dataset
from fit_r2elm import fit_r2elm_on_dataset

iris = fetch_uci_datasets('iris')
liver = fetch_uci_datasets('liver')
X_iris, Y_iris = iris.data, iris.target
X_liver, Y_liver = liver.data, liver.target

scaler = MinMaxScaler((-1,1))
X_iris = scaler.fit_transform(X_iris)

scaler = MinMaxScaler((-1,1))
X_liver = scaler.fit_transform(X_liver)

sig_model = ELM(activation='sigmoid')
rbf_model = ELM(activation='rbf')
sig_model.fit(X_iris,Y_iris)
rbf_model.fit(X_iris,Y_iris)
print "%s: dim: %i, class: %i" % (iris.name, iris.n_dim, iris.n_class)
print "ELM+SIG on iris: %.2f" % accuracy_score(Y_iris, sig_model.predict(X_iris))
print "ELM+RBF on iris: %.2f" % accuracy_score(Y_iris, rbf_model.predict(X_iris))
print
print "%s: dim: %i, class: %i" % (liver.name, liver.n_dim, liver.n_class)
sig_model = ELM(activation='sigmoid')
rbf_model = ELM(activation='rbf')
sig_model.fit(X_liver,Y_liver)
rbf_model.fit(X_liver,Y_liver)
print "ELM+SIG on liver: %.2f" % accuracy_score(Y_liver, sig_model.predict(X_liver))
print "ELM+RBF on liver: %.2f" % accuracy_score(Y_liver, rbf_model.predict(X_liver))

# param_grid = {'h': [50],
#               'beta': [0.04]}
#
# E_grid, E = fit_elm_on_dataset(data=dataset, param_grid_in=param_grid, to_file=False)
#
# print E['config']['params']
# print E['results']['mean_acc']
# print E['monitors']['std']
# print E['monitors']['train_time']
#
# print E_grid['monitors']['grid_time']


# data = fetch_uci_datasets('iris')
# data = shuffle_data(data)
# model = R2ELMLearner(seed=42)
#
# X, Y = data.data, data.target
# model.fit(X, Y)
# print sklearn.metrics.accuracy_score(Y, model.predict(X))
