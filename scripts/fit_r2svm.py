import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import pandas as pd
import os
import pickle
import numpy as np
from fit_models import grid_search
from r2 import R2SVMLearner
from datetime import datetime
from data_api import fetch_uci_datasets
from misc.experiment_utils import get_logger


def main():

    exp_name = 'R2SVM_grid_' + str(datetime.now().time())[:-7]

    params = {'C': [np.exp(i) for i in xrange(-2, 5)],
              'beta': [0.04 * i for i in xrange(1, 6)],
              'depth': [i for i in xrange(2, 11, 3)],
              'fit_c': [None],
              'activation': ['sigmoid'],
              'scale': [True], #[True, False],
              'recurrent': [True, False],
              'use_prev': [True, False],
              'seed': [666]}

    small_datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])
    model = R2SVMLearner()
    logger = get_logger(exp_name, to_file=False)
    results = {}

    for data in small_datasets:
        exp = grid_search(model, data, params, logger=logger, verbose=3)
        results[data.name] = pd.DataFrame.from_dict(exp['monitors'].update(exp['results']))
        print data.name + " done!"

    f = open(os.path.join('./cache/' + exp_name + '.pkl'), 'wb')
    pickle.dump(results, f)
    f.close()

if __name__ == '__main__':
    main()
