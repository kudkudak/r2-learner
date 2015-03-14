import pandas as pd
import os
import pickle
from fit_models import grid_search
from models import R2ELMLearner
from datetime import datetime
from data_api import fetch_uci_datasets
from misc.experiment_utils import get_logger


def main():

    exp_name = 'R2ELM_grid_' + str(datetime.now().time())[:-7]

    params = {'h': [i for i in xrange(20,101,20)],
              'beta': [0.05 * i for i in xrange(1,5)],
              'fit_h': [None], # 'grid', 'random' or None
              'depth': [i for i in xrange(1,11,2)],
              'scale': [True, False],
              'recurrent': [True, False],
              'use_prev': [True, False],
              'seed': [666]}

    small_datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])
    model = R2ELMLearner()
    logger = get_logger(exp_name, to_file=False)
    results = {}

    for data in small_datasets:
        exp = grid_search(model, data, params, logger=logger)
        results[data.name] = pd.DataFrame.from_dict(exp['monitors'].update(exp['results']))
        print data.name + " done!"

    f = open(os.path.join('./cache/' + exp_name + '.pkl'), 'wb')
    pickle.dump(results, f)
    f.close()

if __name__ == '__main__':
    main()


