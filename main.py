
"""
THIS IS A TEST FILE, I KNOW IT'S A MESS
"""

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from data_api import *
from models import R2SVMLearner, R2Lin
import sklearn
from sklearn.grid_search import GridSearchCV
from fit_svm import fit_on_dataset

dataset = fetch_uci_datasets('iris')

param_grid = {'C': [0.1,1],
              'beta': [0.04]}

E_grid, E = fit_on_dataset(dataset, param_grid_in=param_grid, to_file=True)

print E['config']['params']
print E['results']['mean_acc']
print E['monitors']['std']
print E['monitors']['train_time']

print E_grid['monitors']['grid_time']


# iris = fetch_uci_datasets('iris')
# model = R2SVMLearner()
# grid = GridSearchCV(model, {'C': [1,2,3]}, scoring='accuracy', cv=KFold(iris.data.shape[0],shuffle=True))
# grid.fit(iris.data, iris.target)
#
# print grid.grid_scores_[0][0]
# print grid.grid_scores_[0][1]
# print grid.grid_scores_[0][2]

