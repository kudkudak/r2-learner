from data_api import fetch_uci_datasets
from misc.params import *
from misc.config import c
from fit_models import grid_model_on_dataset, test_model_on_dataset
from models import R2ELMLearner, R2SVMLearner, ELM
from sklearn.svm import SVC
import time, cPickle
import os
import pandas as pd

if __name__ == '__main__':

    small_datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])
    medium_datasets = fetch_uci_datasets(['satimage', 'segment'])
    models = ['r2svm', 'r2elm', 'elm+rbf', 'elm+sig', 'svm']
    datasets = small_datasets

    params = [r2svm_params, r2elm_params, elmrbf_params, elmsig_params, svmrbf_params]

    zipped_params = zip(models, params)

    scores = {m: {d.name: 0. for d in small_datasets} for m in models}

    for data in datasets :
        for name, params in zipped_params:
            config = {'model': params, 'grid': grid_config}
            grid_config['cv'] = 5
            E_grid = grid_model_on_dataset(name, data, config)

            file_name = name + '_grid_on_' + data.name + '_' + str(time.time()) + '.exp'
            f = open(os.path.join(c['CACHE_DIR'],file_name), 'w')
            cPickle.dump((E_grid), f)
            f.close()

            best_params = E_grid['results']['best_params']

            if name == 'r2svm':
                model = R2SVMLearner(**best_params)
            elif name == 'r2elm':
                model = R2ELMLearner(**best_params)
            elif name[:3] == 'elm':
                model = ELM(**best_params)
            elif name == 'svm' :
                model = SVC(**best_params)

            scores[name][data.name] = test_model_on_dataset(model, data, best_params)
            print name + " done!"

        print data.name + " done!"

    scores = pd.DataFrame.from_dict(scores, dtype=np.float64)

    file_name = 'models_test_accuracy' + str(time.time()) + '.exp'
    f = open(os.path.join(c['CACHE_DIR'],file_name), 'w')
    cPickle.dump((scores), f)
    f.close()

    pd.options.display.float_format = '{:,.4f}'.format
    print scores
