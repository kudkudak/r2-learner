import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import pandas as pd
import os
import pickle
import numpy as np
from fit_models import grid_search
from elm import ELM
from datetime import datetime
from data_api import fetch_uci_datasets
from misc.experiment_utils import get_logger
from misc.config import c

def main():

    # RBF
    exp_name = 'rbfELM_grid_' + str(datetime.now().time())[:-7]

    rbf_params = {'h': [i for i in xrange(20, 101, 10)],
                     'activation' : ['rbf'],
                     'random_state': [666]}

    datasets = fetch_uci_datasets(['liver', 'iris', 'heart'])
    model = ELM()
    logger = get_logger(exp_name, to_file=False)
    results = {d.name: {} for d in datasets}
    monitors = {d.name: {} for d in datasets}

    for data in datasets:
        exp = grid_search(model, data, rbf_params, logger=logger, verbose=0)
        results[data.name] = exp['results']
        monitors[data.name] = exp['monitors']
        results[data.name].update(monitors[data.name])
        print data.name + " done!"

    ret = pd.DataFrame.from_dict(results)

    # f = open(os.path.join(c["RESULTS_DIR"],exp_name + '.pkl'), 'wb')
    # pickle.dump(results, f)
    # f.close()

    # SIG
    exp_name = 'sigELM_grid_' + str(datetime.now().time())[:-7]

    sig_params = {'h': [i for i in xrange(20, 101, 10)],
                     'activation' : ['sigmoid'],
                     'random_state': [666]}

    logger = get_logger(exp_name, to_file=False)
    results = {d.name: {} for d in datasets}
    monitors = {d.name: {} for d in datasets}

    for data in datasets:
        exp = grid_search(model, data, sig_params, logger=logger, verbose=3)
        results[data.name] = exp['results']
        monitors[data.name] = exp['monitors']
        results[data.name].update(monitors[data.name])
        print data.name + " done!"

    ret = pd.DataFrame.from_dict(results)

    # f = open(os.path.join(c["RESULTS_DIR"],exp_name + '.pkl'), 'wb')
    # pickle.dump(ret, f)
    # f.close()

if __name__ == '__main__':
    main()
