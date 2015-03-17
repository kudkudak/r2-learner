from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from data_api import shuffle_data
from misc.experiment_utils import save_exp
from datetime import datetime
import time
import numpy as np
from r2 import score_all_depths_r2

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


def k_fold(base_model, params, data, n_folds=5, seed=None, store_clf=True, logger=None):

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

    # change it!
    config['experiment_name'] = 'r2svm_' + data.name + '_' +  str(datetime.now().time())[:-7]

    monitors["acc_fold"] = []
    monitors["train_time"] = []
    monitors["test_time"] = []
    monitors["clf"] = []
    monitors["best_depth"] = []

    X, Y = data.data, data.target
    folds = StratifiedKFold(y=Y, n_folds=n_folds, shuffle=True, random_state=seed)
    i = 0
    for train_index, test_index in folds:
        i += 1
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        train_start = time.time()
        model = base_model.set_params(**params)
        model.fit(X_train, Y_train)
        monitors['train_time'].append(time.time() - train_start)

        if store_clf :
            monitors['clf'].append(model)

        test_start = time.time()
        # Y_predicted = model.predict(X_test)
        best_depth, score = score_all_depths_r2(model, X_test, Y_test)
        monitors['test_time'].append(time.time() - test_start)

        monitors['acc_fold'].append(score)
        monitors['best_depth'].append(best_depth)
        #print "Done: %i/%i" % (i, len(folds))

    monitors['acc_fold'] = np.array(monitors['acc_fold'])
    monitors['std'] = monitors['acc_fold'].std()

    monitors['n_dim'] = data.n_dim
    monitors['n_class'] = data.n_class
    monitors['data_name'] = data.name

    results["mean_acc"] = monitors["acc_fold"].mean()

    if logger is not None :
        logger.info(results)
        logger.info(monitors)

    # save_exp(experiment)
    return experiment


def nk_folds(model, params, data, n=50, n_folds=10, n_jobs=4):

    model.set_params(params)
    scores = []
    for _ in xrange(n) :
        data = shuffle_data(data)
        scores.append(cross_val_score(estimator=model, X=data.data, y=data.target, scoring='accuracy', cv=n_folds, n_jobs=n_jobs).mean())

    return np.mean(scores), np.std(scores)