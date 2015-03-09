from models import R2ELMLearner
from misc.experiment_utils import get_exp_logger

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

import time
import numpy as np

# TODO: make a functions for running k-fold n-times

def fit_r2elm_grid(data, config, logger=None):
    """
    Fits a GridSearchCV class from scikit-learn
    :param data: dict-like with fields: name, data, target
    :param config: dictionary with parameters specific for a grid search
    :param logger: logger class
    :return: dictionary Experiment with gird results
    """
    assert hasattr(data, 'name')
    assert hasattr(data, 'data')
    assert hasattr(data, 'target')
    assert ['experiment_type', 'param_grid','scoring', 'cv', 'refit', 'store_clf'] <= config.keys()
    assert config['experiment_type'] == 'grid'
    ### Prepare result holders ###
    results = {}
    monitors = {}

    E = {"config": config, "results": results, "monitors": monitors}

    model = R2ELMLearner()
    cv_grid = GridSearchCV(model, param_grid=config['param_grid'], scoring=config['scoring'], n_jobs=-1, cv=config['cv'])

    X = data.data
    Y = data.target

    start_time = time.time()
    cv_grid.fit(X, Y)
    monitors['grid_time'] = time.time() - start_time

    results['best_params'] = cv_grid.best_params_
    results['best_score'] = cv_grid.best_score_
    if config['refit'] :
        results['best_cls'] = R2ELMLearner(**cv_grid.best_params_).fit(X,Y)
    else :
        results['best_cls'] = cv_grid.best_estimator_

    monitors['fold_params'] = [ s[0] for s in cv_grid.grid_scores_ ]
    monitors['mean_fold_scores'] = [s[1] for s in cv_grid.grid_scores_]
    monitors['std_fold_scores'] = [np.std(s[2]) for s in cv_grid.grid_scores_]
    monitors['best_std'] = [ np.std(s[2]) for s in cv_grid.grid_scores_ if s[1] == cv_grid.best_score_ ]
    monitors['data_name'] = data.name

    if config['store_clf'] :
        monitors['clf'] = cv_grid

    if logger is not None :
        logger.info(results)
        logger.info(monitors)

    return E

def fit_r2elm(data, config, logger=None) :
    """
    Fits R2SVMLearner class on
    :param data: dict-like with fields: name, data, target
    :param config: dictionary with parameters specific for a k-fold fitting
    :param logger: logger class
    :return: dictionary Experiment with cross validation results results
    """
    assert hasattr(data, 'name')
    assert hasattr(data, 'data')
    assert hasattr(data, 'target')
    assert ['experiment_type', 'n_folds', 'fold_seed', 'params', 'store_clf'] <= config.keys()
    assert config['experiment_type'] == 'k-fold'

    ### Prepare result holders ###b
    results = {}
    monitors = {}
    E = {"config": config, "results": results, "monitors": monitors}

    monitors["acc_fold"] = []
    monitors["train_time"] = []
    monitors["test_time"] = []
    monitors["cm"] = [] # confusion matrix
    monitors["clf"] = []

    X, Y = data.data, data.target
    folds = KFold(n=X.shape[0], n_folds=config['n_folds'], shuffle=True, random_state=config['fold_seed'])
    for train_index, test_index in folds :
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        model = R2ELMLearner(**config['params'])
        train_start = time.time()
        model.fit(X_train, Y_train)
        monitors['train_time'].append(time.time() - train_start)

        if config['store_clf'] :
            monitors['clf'].append(model)

        test_start = time.time()
        Y_predicted = model.predict(X_test)
        monitors['test_time'] = time.time() - test_start

        monitors['acc_fold'].append(accuracy_score(Y_test, Y_predicted))
        monitors['cm'].append(confusion_matrix(Y_test, Y_predicted))

    monitors['acc_fold'] = np.array(monitors["acc_fold"])
    monitors['std'] = monitors['acc_fold'].std()
    monitors['n_dim'] = data.n_dim
    monitors['n_class'] = data.n_class
    monitors['data_name'] = data.name

    results['mean_acc'] = monitors['acc_fold'].mean()

    if logger is not None :
        logger.info(results)
        logger.info(monitors)

    return E


def fit_r2elm_grid_default(data):
    """Fits a exhausting grid search with default parameters on a data set, return dictionary with results"""

    config = {'experiment_type': 'grid',
                  'param_grid': default_r2elm_grid_parameters(),
                  'n_fold': 3,
                  'scoring': 'accuracy',
                  'store_clf': False,
                  'seed': None,
                  'experiment_name': 'R2ELM default grid experiment on ' + data.name}

    assert hasattr(data, 'name')
    assert hasattr(data, 'data')
    assert hasattr(data, 'target')

    return  fit_r2elm_grid(config, data.data, data.target)

def fit_r2elm_on_dataset(data, param_grid_in=None, grid_config_in=None, fold_config_in=None, to_file=False):

    param_grid = default_r2elm_grid_parameters()
    if param_grid_in is not None :
        param_grid.update(param_grid_in)

    fold_seed = np.random.randint(0, np.iinfo(np.int32).max)
    random_state = np.random.RandomState(fold_seed)

    grid_config = {'experiment_name': 'R2ELM grid_search_on_' + data.name + str(time.time()),
                   'experiment_type': 'grid',
                   'refit': True,
                   'scoring': 'accuracy',
                   'fold_seed': fold_seed,
                   'cv': KFold(n=data.data.shape[0], shuffle=True, n_folds=3, random_state=random_state),
                   'store_clf': False,
                   'param_grid': param_grid}

    if grid_config_in is not None :
        grid_config.update(grid_config_in)

    logger = get_exp_logger(grid_config, to_file=to_file)

    E_grid = fit_r2elm_grid(data, grid_config, logger)
    params = E_grid['results']['best_params']

    fold_config = {'experiment_name': 'R2ELM k-fold_on_' + data.name + str(time.time()),
                   'experiment_type': 'k-fold',
                   'n_folds': 5,
                   'fold_seed': E_grid['config']['fold_seed'],
                   'store_clf': True,
                   'params': params}

    if fold_config_in is not None :
        fold_config.update(fold_config_in)

    logger = get_exp_logger(fold_config, to_file=to_file)

    return E_grid, fit_r2elm(data, fold_config, logger)

def default_r2elm_grid_parameters() :
    return { 'h': [i for i in xrange(10,60,10)],
             'beta': [0.05 * i for i in xrange(0,7)],
             'fit_h': None, # 'grid', 'random' or None
             'depth': [5],
             'scale': [True],
             'recurrent': [True],
             'use_prev': [True],
             'seed': [None]}
