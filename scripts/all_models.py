from fit_models import fit_model_on_dataset
from data_api import fetch_uci_datasets
import numpy as np
import cPickle

if __name__ == '__main__':

    datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])
    models = ['r2svm', 'r2elm', 'elm', 'elm', 'svm']
    seed = 42
    grid_config = {'refit': True, 'scoring': 'accuracy', 'fold_seed': seed, 'store_clf': True}
    fold_config = {'n_folds': 5, 'fold_seed': seed, 'store_clf': True}

    r2svm_params = {'C': [np.exp(i) for i in xrange(-2,6)],
                    'beta': [0.05 * i for i in xrange(0,7)],
                    'depth': [5,10],
                    'fit_c': ['random'],
                    'scale': [True],
                    'recurrent': [True],
                    'use_prev': [True, False],
                    'seed': [seed]}

    r2elm_params = {'h': [i for i in xrange(10,60,10)],
                    'beta': [0.05 * i for i in xrange(0,7)],
                    'fit_h': [None, 'random'], # 'grid', 'random' or None
                    'depth': [5,10],
                    'scale': [True],
                    'recurrent': [True],
                    'use_prev': [True, False],
                    'seed': [seed]}

    elmrbf_params = {'h': [i for i in xrange(50, 100, 10)],
                     'activation' : ['rbf'],
                     'seed': [seed]}

    elmsig_params = {'h': [i for i in xrange(50, 100, 10)],
                     'activation' : ['sigmoid'],
                     'seed': [seed]}

    svmrbf_params = {'C': [np.exp(i) for i in xrange(-2,6)],
                     'kernel': ['rbf'],
                     'gamma': [np.exp(i) for i in xrange(-10,11,2)],
                     'class_weight': ['auto']}

    params = [r2svm_params, r2svm_params, elmrbf_params, elmsig_params, svmrbf_params]

    for data in datasets :
        for model, params in zip(models, params) :
            config = {'model': params, 'grid': grid_config, 'fold': fold_config}
            E_grid, E_detailed = fit_model_on_dataset(model, data, config)

            file_name = model + '_' + data.name + '.exp'
            f = open(file_name, 'w')
            cPickle.dump((E_grid, E_detailed), f)
            f.close()
            print model + " done!"

