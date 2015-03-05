
"""
THIS IS A TEST FILE, I KNOW IT'S A MESS
"""

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from data_api import *
from models import R2SVMLearner, R2Lin
import sklearn
from sklearn.grid_search import GridSearchCV
from fit_svm import fit_grid, fit_on_dataset

grid_config = {'experiment_type': 'grid',
                  'param_grid': { 'C': [10**i for i in xrange(0,3)],
                                   'beta': [0.02 * i for i in xrange(1,2)],
                                   'depth': [5],
                                   'scale': [True, False],
                                   'recurrent': [True, False],
                                   'use_prev': [True, False]},
                  'n_fold': 3,
                  'scoring': 'accuracy',
                  'store_clf': False,
                  'seed': None,
                  'experiment_name': 'my fabulous experiment'}

fold_config = {         'experiment_name': 'k-fold experiment',
                        'experiment_type': 'k-fold',
                        'store_clf': False,
                        'n_folds': 3,
                        'fold_seed': None}

dataset = fetch_uci_datasets('iris')

E_grid, E = fit_on_dataset(dataset, grid_config, fold_config)

print E['config']['params']
print E['config']['dataset']
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

