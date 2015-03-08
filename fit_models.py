from fit_r2elm import fit_r2elm_on_dataset
from fit_r2svm import fit_r2svm_on_dataset
from fit_elms import fit_elm_on_dataset
from fit_svms import fit_svc_on_dataset


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
    elif name == 'elm':
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