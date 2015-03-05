"""
First version, it's not so nice, but should be efficient
I know some things needs refactoring, don't shout on me
    - Igor
"""


from models import R2SVMLearner
from data_api import fetch_uci_datasets
from misc.experiment_utils import *
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
import time

# TODO: do something with those 2 different config types

def fit_sk_grid(config, X, Y):
    # TODO: more asserts
    assert config['experiment_type'] == 'grid'
    ### Prepare result holders ###b
    results = {}
    monitors = {}

    E = {"config": config, "results": results, "monitors": monitors}

    model = R2SVMLearner()
    cv_grid = GridSearchCV(model, param_grid=config['param_grid'], scoring=config['scoring'], n_jobs=-1)

    start_time = time.time()
    cv_grid.fit(X, Y)
    monitors['grid_time'] = time.time() - start_time

    results['best_score'] = cv_grid.best_score_
    results['best_cls'] = cv_grid.best_estimator_
    results['best_params'] = cv_grid.best_params_

    monitors['mean_fold_scores'] = [s[1] for s in cv_grid.grid_scores_]
    monitors['std_fold_scores'] = [np.std(s[2]) for s in cv_grid.grid_scores_]
    monitors['params_fold'] = [s[0] for s in cv_grid.grid_scores_]
    monitors['best_std'] = [ np.std(s[2]) for s in cv_grid.grid_scores_ if s[1] == cv_grid.best_score_ ]

    if config['store_clf'] :
        monitors['clf'] = cv_grid

    return E

def fit_r2svm(config, X, Y) :

    # TODO: more asserts
    assert config['experiment_type'] == 'k-fold'

    ### Prepare result holders ###b
    results = {}
    monitors = {}
    E = {"config": config, "results": results, "monitors": monitors}

    monitors["acc_fold"] = []
    # monitors["mcc_fold"] = []
    monitors["train_time"] = []
    monitors["test_time"] = []
    monitors["cm"] = [] # confusion matrix
    monitors["clf"] = []

    n_folds = config['n_folds'] if config['n_folds'] is not None else 3
    folds = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=config['fold_seed'])
    for train_index, test_index in folds :
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        model = R2SVMLearner(**config['params'])
        train_start = time.time()
        model.fit(X_train, Y_train)
        monitors['train_time'].append(time.time() - train_start)

        if config['store_clf'] :
            monitors['clf'].append(model)

        test_start = time.time()
        Y_predicted = model.predict(X_test)
        monitors['test_time'] = time.time() - test_start

        monitors['acc_fold'].append(accuracy_score(Y_test, Y_predicted))
        # TODO: if this by n_class
        # monitors['mcc_fold'].append(matthews_corrcoef(Y_test, Y_predicted))
        monitors['cm'].append(confusion_matrix(Y_test, Y_predicted))

    monitors["acc_fold"] = np.array(monitors["acc_fold"])
    #monitors["mcc_fold"] = np.array(monitors["mcc_fold"])
    monitors['std'] = monitors['acc_fold'].std()

    results["mean_acc"] = monitors["acc_fold"].mean()
    #results["mean_mcc"] = monitors["mcc_fold"].mean()


    return E


def fit_grid(data, config_in=None):
    """Fits a exhausting grid search on a data set, return dictionary with results"""

    if config_in is None :

        # TODO: move this somewhere else as default grid config
        config = {'experiment_type': 'grid',
                  'param_grid': { 'C': [10**i for i in xrange(-10,11)],
                                   'beta': [0.02 * i for i in xrange(0,6)],
                                   'depth': [7],
                                   'scale': [True, False],
                                   'recurrent': [True, False],
                                   'use_prev': [True, False]},
                  'n_fold': 3,
                  'scoring': 'accuracy',
                  'store_clf': False,
                  'seed': None,
                  'experiment_name': 'default grid experiment on ' + data.name}
    else :
        assert config_in['experiment_type'] == 'grid'
        config = config_in


    assert hasattr(data, 'name')
    assert hasattr(data, 'data')
    assert hasattr(data, 'target')

    config['dataset'] = data.name

    return  fit_sk_grid(config, data.data, data.target)

def fit_on_dataset(data, grid_config=None, fold_config=None):

    E_grid = fit_grid(data, grid_config)
    params = E_grid['results']['best_params']

    # TODO: move this somewhere else as default fold config
    if fold_config is None :
        fold_config = { 'experiment_name': 'k-fold testing on a ' + data.name,
                        'experiment_type': 'k-fold',
                        'store_clf': False,
                        'n_folds': grid_config['n_fold'],
                        'fold_seed': None}
    elif 'dataset' in fold_config.keys() :
        if fold_config['dataset'] != E_grid['config']['dataset']:
            print "Ignoring dataset name in fold config"

    fold_config['params'] = params
    fold_config['dataset'] = E_grid['config']['dataset']


    return E_grid, fit_r2svm(fold_config, data.data, data.target)
