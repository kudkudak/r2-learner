from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer
from data_api import shuffle_data
from misc.experiment_utils import save_exp, get_exp_logger, shorten_params, exp_done
from datetime import datetime
import time
import numpy as np
import sys
from r2 import score_all_depths_r2, _r2_compress_model
import scipy
from sklearn.base import clone
from copy import copy

def grid_search(model, data, param_grid, logger=None, scoring='accuracy', store_clf=False, n_jobs=8,
                seed=None, more=False, n_folds=5, verbose=0):
    """
    :param model:       initialized model
    :param data:        dict-like with data, targetm name fields
    :param param_grid:  params for grid search
    :param logger:      logger class
    :param scoring:     scoring function, string or callable
    :param store_clf:   save best classifier in monitors
    :param n_jobs:      number of jobs to run on
    :param seed:        seed for shuffling data
    :param more:        save additional info in monitors
    :param n_folds:     number of folds
    :return:            dict with config, results and monitors fields
    """

    assert hasattr(data, 'name')
    assert hasattr(data, 'data')
    assert hasattr(data, 'target')

    results = {}
    monitors = {}
    config = {}
    experiment = {"config": config, "results": results, "monitors": monitors}

    config['n_dim'] = data.n_dim
    config['n_class'] = data.n_class
    config['data_name'] = data.name
    config['scoring'] = scoring
    config['store_clf'] = store_clf
    config['n_jobs'] = n_jobs
    config['seed'] = seed
    config['more'] = more
    config['n_folds'] = n_folds

    X = data.data
    Y = data.target

    cv = StratifiedKFold(y=Y, n_folds=n_folds, shuffle=True, random_state=seed)
    cv_grid = GridSearchCV(model, param_grid=param_grid, scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=verbose)

    start_time = time.time()
    cv_grid.fit(X, Y)
    monitors['grid_time'] = time.time() - start_time

    results['best_params'] = cv_grid.best_params_
    results['best_score'] = cv_grid.best_score_

    if more:
        monitors['fold_params'] = [ s[0] for s in cv_grid.grid_scores_ ]
        monitors['mean_fold_scores'] = [s[1] for s in cv_grid.grid_scores_]
        monitors['std_fold_scores'] = [np.std(s[2]) for s in cv_grid.grid_scores_]

    monitors['best_std'] = [ np.std(s[2]) for s in cv_grid.grid_scores_ if s[0] == cv_grid.best_params_ ]

    if store_clf:
        monitors['clf'] = cv_grid

    if logger is not None:
        logger.info(results)
        logger.info(monitors)

    save_exp(experiment)


def k_fold(base_model, params, data, exp_name, model_name,  n_folds=5, seed=None, store_clf=False, log=True, n_tries=3, save_model=True, all_layers=True):

    assert hasattr(data, 'name')
    assert hasattr(data, 'data')
    assert hasattr(data, 'target')

    if seed is None:
        seed = params['seed']

    results = {}
    monitors = {}
    config = {}
    experiment = {"config": config, "results": results, "monitors": monitors}

    config['n_folds'] = n_folds
    config['seed'] = seed
    config['store_clf'] = store_clf
    config['params'] = params

    short_params = shorten_params(params)

    # change it!
    config['experiment_name'] = exp_name + '_' + model_name + '_' + data.name + '_' + short_params
    dir_name = exp_name + '_' + model_name + '_' + data.name

    if save_model and exp_done(experiment, dir_name):
        print "exp already done"
        return

    monitors["fold_scores"] = []
    monitors["train_time"] = []
    monitors["test_time"] = []
    monitors["clf"] = []

    if log:
        logger = get_exp_logger(config, dir_name, to_file=True, to_std=False)

    X, Y = data.data, data.target
    folds = StratifiedKFold(y=Y, n_folds=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in folds:

        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        fold_scores = []
        fold_train_times = []
        fold_test_times = []

        for seed_bias in xrange(n_tries):
            fold_params = copy(params)
            fold_params['seed'] += seed_bias
            train_start = time.time()
            model = base_model(**fold_params)
            model.fit(X_train, Y_train)
            fold_train_times.append(time.time() - train_start)

            test_start = time.time()

            if all_layers:
                fold_scores.append(score_all_depths_r2(model, X_test, Y_test))
            else:
                Y_pred = model.predict(X_test)
                fold_scores.append(accuracy_score(Y_test, Y_pred))

            fold_test_times.append(time.time() - test_start)

            if store_clf :
                monitors['clf'].append(_r2_compress_model(model))

        monitors['train_time'].append(fold_train_times)
        monitors['test_time'].append(fold_test_times)
        monitors['fold_scores'].append(np.mean(np.array(fold_scores), axis=0))

    monitors['n_dim'] = data.n_dim
    monitors['n_class'] = data.n_class
    monitors['data_name'] = data.name

    monitors['fold_scores'] = np.array(monitors['fold_scores'])

    if all_layers:
        results['best_depth'] = np.argmax(np.mean(monitors['fold_scores'], axis=0)) + 1
        results['mean_acc'] = np.max(np.mean(monitors['fold_scores'], axis=0))
        # results['std'] = monitors['fold_scores'][results['best_depth'] - 1, :].std()
    else:
        results['mean_acc'] = monitors['fold_scores'].mean()
        results['std'] = monitors['fold_scores'].std()
        results['best_depth'] = params['depth']

    if log:
        logger.info(config)
        logger.info(results)
        logger.info(monitors)

    if save_model:
        save_exp(experiment, dir_name)

    return experiment


def nk_folds(model, params, data, n=50, n_folds=10, n_jobs=4):

    model.set_params(params)
    scores = []
    for _ in xrange(n) :
        data = shuffle_data(data)
        scores.append(cross_val_score(estimator=model, X=data.data, y=data.target, scoring='accuracy', cv=n_folds, n_jobs=n_jobs).mean())

    return np.mean(scores), np.std(scores)

def extern_k_fold(base_model, params, data, exp_name, model_name, n_folds=5, seed=666, store_clf=False, log=True, save_model=True):

    assert hasattr(data, 'name')
    assert hasattr(data, 'data')
    assert hasattr(data, 'target')

    results = {}
    monitors = {}
    config = {}
    experiment = {"config": config, "results": results, "monitors": monitors}

    config['n_folds'] = n_folds
    config['seed'] = seed
    config['store_clf'] = store_clf
    config['params'] = params

    short_params = shorten_params(params)

    config['experiment_name'] = exp_name + '_' + model_name + '_' + data.name + '_' + short_params
    dir_name = exp_name + '_' + model_name + '_' + data.name

    if save_model and exp_done(experiment, dir_name):
        print "exp already done"
        return

    monitors["acc_fold"] = []
    monitors["train_time"] = []
    monitors["test_time"] = []
    monitors["clf"] = []

    if log:
        logger = get_exp_logger(config, dir_name, to_file=True, to_std=False)

    Y = data.target
    X = MinMaxScaler((-1,1)).fit_transform(data.data)

    folds = StratifiedKFold(y=Y, n_folds=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in folds:
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        train_start = time.time()
        model = clone(base_model).set_params(**params)
        model.fit(X_train, Y_train)
        train_time = time.time() - train_start

        test_start = time.time()
        Y_pred = model.predict(X_test)
        test_time = time.time() - test_start
        score = accuracy_score(Y_test, Y_pred)

        if store_clf :
            monitors['clf'].append(model)

        monitors['train_time'].append(train_time)
        monitors['test_time'].append(test_time)
        monitors['acc_fold'].append(score)

    monitors['acc_fold'] = np.array(monitors['acc_fold'])
    monitors['std'] = monitors['acc_fold'].std()

    monitors['n_dim'] = data.n_dim
    monitors['n_class'] = data.n_class
    monitors['data_name'] = data.name

    results["mean_acc"] = monitors["acc_fold"].mean()

    if log:
        logger.info(config)
        logger.info(results)
        logger.info(monitors)

    # UNCOMMENT FOR REAL TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if save_model:
        save_exp(experiment, dir_name)

    return experiment
