from fit_models import fit_model_on_dataset, test_model_on_dataset
from data_api import fetch_uci_datasets
from misc.params import *
import os
from misc.config import c
import cPickle
import time
import pandas as pd


if __name__ == '__main__':

    small_datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])
    medium_datasets = fetch_uci_datasets(['satimage', 'segment'])

    models = ['r2svm', 'r2elm', 'elm+rbf', 'elm+sig', 'svm']

    params = [r2svm_params, r2elm_params, elmrbf_params, elmsig_params, svmrbf_params]

    zipped_params = zip(models, params)

    results = {d.name: {m: {} for m in models} for d in small_datasets}

    for data in small_datasets :
        for name, params in zipped_params:
            config = {'model': params, 'grid': grid_config, 'fold': fold_config}
            E_grid, E_detailed = fit_model_on_dataset(name, data, config)
            file_name = name + '_' + data.name + '_' + str(time.time()) + '.exp'

            results[data.name][name].update(E_detailed['monitors'])
            results[data.name][name].update(E_grid['monitors'])
            results[data.name][name].update(E_grid['results'])
            results[data.name][name]['n_dim'] = data.n_dim
            results[data.name][name]['n_class'] = data.n_class
            m, s = test_model_on_dataset(name, E_grid['results']['best_params'], data)
            results[data.name][name]['50x5-fold_mean'] = m
            results[data.name][name]['50x5-fold_std'] = s

            f = open(os.path.join(c['CACHE_DIR'],file_name), 'w')
            cPickle.dump((E_grid, E_detailed), f)
            f.close()

            del results[data.name][name]['clf']
            del results[data.name][name]['data_name']
            del results[data.name][name]['n_dim']
            del results[data.name][name]['n_class']
            del results[data.name][name]['best_cls']
            print name + " done!"
        print data.name + " done!"

    ret = { key: pd.DataFrame.from_dict(val) for key, val in results.iteritems() }

    file_name ='all_models_iris_liver_heart.res'
    f = open(os.path.join(c['CACHE_DIR'],file_name), 'w')
    cPickle.dump(ret, f)
    f.close()

    print ret

    # for data in datasets :
    #     config = {'model': r2svm_params_fit_c, 'grid': grid_config, 'fold': fold_config}
    #     E_grid, E_detailed = fit_model_on_dataset('r2svm', data, config)
    #
    #     file_name = 'r2svm_fit_c' + '_' + data.name + '_' + str(time.time()) + '.exp'
    #     f = open(file_name, 'w')
    #     cPickle.dump((E_grid, E_detailed), f)
    #     f.close()
    #     print "r2svm with fit_c" + data.name + " done!"