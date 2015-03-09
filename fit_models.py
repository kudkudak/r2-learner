from fit_r2elm import fit_r2elm_on_dataset, fit_r2elm_grid
from fit_r2svm import fit_r2svm_on_dataset, fit_r2svm_grid
from fit_elms import fit_elm_on_dataset, fit_elm_grid
from fit_svms import fit_svc_on_dataset, fit_svc_grid
from models import R2ELMLearner, R2SVMLearner, ELM
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from data_api import shuffle_data
import time
from misc.experiment_utils import get_exp_logger
import numpy as np

def fit_model_on_dataset(name, data, params):

    E_grid, E_detailed = None, None

    if name == 'r2svm':
        E_grid, E_detailed = fit_r2svm_on_dataset(data,
                                                  param_grid_in=params['model'],
                                                  grid_config_in=params['grid'],
                                                  fold_config_in=params['fold'],
                                                  to_file=True)
    elif name == 'r2elm':
        E_grid, E_detailed = fit_r2elm_on_dataset(data,
                                                  param_grid_in=params['model'],
                                                  grid_config_in=params['grid'],
                                                  fold_config_in=params['fold'],
                                                  to_file=True)
    elif name[:3] == 'elm':
        E_grid, E_detailed = fit_elm_on_dataset(data,
                                                param_grid_in=params['model'],
                                                grid_config_in=params['grid'],
                                                fold_config_in=params['fold'],
                                                to_file=True)
    elif name == 'svm' :
        E_grid, E_detailed = fit_svc_on_dataset(data,
                                                param_grid_in=params['model'],
                                                grid_config_in=params['grid'],
                                                fold_config_in=params['fold'],
                                                to_file=True)

    return E_grid, E_detailed

def grid_model_on_dataset(name, data, params):

    E_grid, E_detailed = None, None
    config = params['grid']
    config['param_grid'] = params['model']
    config['experiment_name'] = name + "_grid_on_" + data.name + '_' + str(time.time())
    logger = get_exp_logger(config, to_file=True)

    if name == 'r2svm':
        E_grid = fit_r2svm_grid(data, config, logger)
    elif name == 'r2elm':
        E_grid = fit_r2elm_grid(data,config, logger)
    elif name[:3] == 'elm':
        E_grid= fit_elm_grid(data, config, logger)
    elif name == 'svm' :
        E_grid = fit_svc_grid(data, config, logger)

    return E_grid

def test_model_on_dataset(name, params, data, n=50, n_folds=5):

    if name == 'r2svm':
        model = R2SVMLearner(**params)
    elif name == 'r2elm':
        model = R2ELMLearner(**params)
    elif name[:3] == 'elm':
        model = ELM(**params)
    elif name == 'svm' :
        model = SVC(**params)

    scores = []
    for _ in xrange(n) :
        data = shuffle_data(data)
        scores.append(cross_val_score(estimator=model, X=data.data, y=data.target, scoring='accuracy', cv=n_folds, n_jobs=-1).mean())

    return np.mean(scores), np.std(scores)