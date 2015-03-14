import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import pandas as pd
import os
import pickle
import numpy as np
from fit_models import grid_search
from models import ELM
from datetime import datetime
from data_api import fetch_uci_datasets
from misc.experiment_utils import get_logger


def main():

    # RBF
    exp_name = 'rbfELM_grid_' + str(datetime.now().time())[:-7]

    rbf_params = {'h': [i for i in xrange(20, 201, 10)],
                     'activation' : ['rbf'],
                     'seed': [666]}

    small_datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])
    model = ELM()
    logger = get_logger(exp_name, to_file=False)
    results = {}

    for data in small_datasets:
        exp = grid_search(model, data, rbf_params, logger=logger, verbose=8)
        results[data.name] = pd.DataFrame.from_dict(exp['monitors'].update(exp['results']))
        print 'ebfELM on ' + data.name + " done!"

    f = open(os.path.join('./cache/' + exp_name + '.pkl'), 'wb')
    pickle.dump(results, f)
    f.close()

    # SIG
    exp_name = 'sigELM_grid_' + str(datetime.now().time())[:-7]

    sig_params = {'h': [i for i in xrange(20, 201, 10)],
                     'activation' : ['sigmoid'],
                     'seed': [666]}

    logger = get_logger(exp_name, to_file=False)
    results = {}

    for data in small_datasets:
        exp = grid_search(model, data, sig_params, logger=logger)
        results[data.name] = pd.DataFrame.from_dict(exp['monitors'].update(exp['results']))
        print 'sigELM on ' + data.name + " done!"

    f = open(os.path.join('./cache/' + exp_name + '.pkl'), 'wb')
    pickle.dump(results, f)
    f.close()

if __name__ == '__main__':
    main()
