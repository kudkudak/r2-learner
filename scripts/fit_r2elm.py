import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pickle
from fit_models import grid_search
from r2 import R2ELMLearner
from datetime import datetime
from data_api import fetch_small_datasets, fetch_uci_datasets, fetch_medium_datasets
from misc.experiment_utils import get_logger
from misc.config import c

def main():

    exp_name = 'R2ELM_grid_' + str(datetime.now().time())[:-7]

    params = {'h': [i for i in xrange(20,201,20)],
              'beta': [0.05 * i for i in xrange(1,5)],
              'fit_h': [None], # 'grid', 'random' or None
              'depth': [i for i in xrange(2,10,3)],
              'scale': [False],
              'recurrent': [True],
              'use_prev': [False],
              'seed': [666]}

    datasets = fetch_medium_datasets()
    model = R2ELMLearner()
    logger = get_logger(exp_name, to_file=False)
    results = {d.name: {} for d in datasets}
    monitors = {d.name: {} for d in datasets}

    for data in datasets:
        exp = grid_search(model, data, params, logger=logger, verbose=1)
        results[data.name] = exp['results']
        monitors[data.name] = exp['monitors']
        results[data.name].update(monitors[data.name])
        print data.name + " done!"

    ret = pd.DataFrame.from_dict(results)

    f = open(os.path.join(c["RESULTS_DIR"],exp_name + '.pkl'), 'wb')
    pickle.dump(ret, f)
    f.close()

if __name__ == '__main__':
    main()


