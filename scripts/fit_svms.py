import pandas as pd
import os
import pickle
import numpy as np
from fit_models import grid_search
from sklearn.svm import SVC
from datetime import datetime
from data_api import fetch_small_datasets
from misc.experiment_utils import get_logger
from misc.config import c

def main():

    type = 'small_' # small, medium, large
    exp_name = 'rbfSVM_grid_' + type + str(datetime.now().time())[:-7]

    params = {'C': [np.exp(i) for i in xrange(-2,6)],
              'kernel': ['rbf'],
              'gamma': [np.exp(i) for i in xrange(-10,11)],
              'class_weight': ['auto']}

    datasets = fetch_small_datasets()
    model = SVC()
    logger = get_logger(exp_name, to_file=False)
    results = {d.name: {} for d in datasets}
    monitors = {d.name: {} for d in datasets}

    for data in datasets:
        exp = grid_search(model, data, params, logger=logger, verbose=3)
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
