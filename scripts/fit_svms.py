import pandas as pd
import os
import pickle
import numpy as np
from fit_models import grid_search
from sklearn.svm import SVC
from datetime import datetime
from data_api import fetch_uci_datasets
from misc.experiment_utils import get_logger


def main():

    exp_name = 'rbfSVM_grid_' + str(datetime.now().time())[:-7]

    params = {'C': [np.exp(i) for i in xrange(-2,6)],
              'kernel': ['rbf'],
              'gamma': [np.exp(i) for i in xrange(-10,11)],
              'class_weight': ['auto']}

    small_datasets = fetch_uci_datasets(['iris', 'liver', 'heart'])
    model = SVC()
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
