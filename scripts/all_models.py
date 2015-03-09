from fit_models import fit_model_on_dataset
from data_api import fetch_uci_datasets
import numpy as np
import cPickle
import time

if __name__ == '__main__':

    datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])#, 'satimage', 'segment'])
    models = ['r2svm', 'r2elm', 'elm+rbf', 'elm+sig', 'svm']
    seed = 42
    grid_config = {'refit': False, 'scoring': 'accuracy', 'fold_seed': seed, 'store_clf': True}
    fold_config = {'n_folds': 5, 'fold_seed': seed, 'store_clf': True}

    r2svm_params = {'C': [np.exp(i) for i in xrange(-2,6)],
                    'beta': [0.05 * i for i in xrange(0,5)],
                    'depth': [i for i in xrange(1,11)],
                    'fit_c': [None],
                    'scale': [True],
                    'recurrent': [True],
                    'use_prev': [True, False],
                    'seed': [seed]}

    r2elm_params = {'h': [i for i in xrange(20,201,10)],
                    'beta': [0.1],
                    'fit_h': [None], # 'grid', 'random' or None
                    'depth': [i for i in xrange(1,11)],
                    'scale': [True],
                    'recurrent': [True],
                    'use_prev': [True, False],
                    'seed': [seed]}

    elmrbf_params = {'h': [i for i in xrange(20, 201, 10)],
                     'activation' : ['rbf'],
                     'seed': [seed]}

    elmsig_params = {'h': [i for i in xrange(20, 201, 10)],
                     'activation' : ['sigmoid'],
                     'seed': [seed]}

    svmrbf_params = {'C': [np.exp(i) for i in xrange(-2,6)],
                     'kernel': ['rbf'],
                     'gamma': [np.exp(i) for i in xrange(-10,11,2)],
                     'class_weight': ['auto']}

    r2svm_params_fit_c = {'C': [1],
                    'beta': [0.05 * i for i in xrange(0,5)],
                    'depth': [5,10],
                    'fit_c': ['random'],
                    'scale': [True],
                    'recurrent': [True],
                    'use_prev': [True, False],
                    'seed': [seed]}

    params = [r2svm_params, r2elm_params, elmrbf_params, elmsig_params, svmrbf_params]

    zipped_params = zip(models, params)

    for data in datasets :
        for model, params in zipped_params:
            config = {'model': params, 'grid': grid_config, 'fold': fold_config}
            E_grid, E_detailed = fit_model_on_dataset(model, data, config)

            file_name = model + '_' + data.name + '_' + str(time.time()) + '.exp'
            f = open(file_name, 'w')
            cPickle.dump((E_grid, E_detailed), f)
            f.close()
            print model + " done!"
        print data.name + " done!"

    # for data in datasets :
    #     config = {'model': r2svm_params_fit_c, 'grid': grid_config, 'fold': fold_config}
    #     E_grid, E_detailed = fit_model_on_dataset('r2svm', data, config)
    #
    #     file_name = 'r2svm_fit_c' + '_' + data.name + '_' + str(time.time()) + '.exp'
    #     f = open(file_name, 'w')
    #     cPickle.dump((E_grid, E_detailed), f)
    #     f.close()
    #     print "r2svm with fit_c" + data.name + " done!"