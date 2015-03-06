from data_api import fetch_uci_datasets
from fit_svm import fit_on_dataset
import cPickle
import time

### Using previous data VS modifying original data
datasets = fetch_uci_datasets()

param_grid = { 'use_prev': [True] }
grid_config = fold_config = {'experiment_name': 'use_prev test', # use different names for different log files
                             'fit_c': False} # set fit_c=True for better but way longer test
grid_list = []
fold_list = []

# Run on the data sets with: use_prev=True
for data in datasets :
    E_grid, E_fold = fit_on_dataset(data,
                                    param_grid_in=param_grid,
                                    grid_config_in=grid_config,
                                    fold_config_in=fold_config,
                                    to_file=True)
    grid_list.append(E_grid)
    fold_list.append(E_fold)

file_name = 'use_prev=true-' + time.time() + '.exp'
f = open(file_name, 'w')
cPickle.dump(zip(grid_list, fold_list), f)
f.close()

grid_params = { 'use_prev': [False] }

grid_list = []
fold_list = []

# Run on the data sets with: use_prev=False
for data in datasets :
    E_grid, E_fold = fit_on_dataset(data,
                                    param_grid_in=param_grid,
                                    grid_config_in=grid_config,
                                    fold_config_in=fold_config,
                                    to_file=True)
    grid_list.append(E_grid)
    fold_list.append(E_fold)

file_name = 'use_prev=false-' + time.time() + '.exp'
f = open(file_name, 'w')
cPickle.dump(zip(grid_list, fold_list), f)
f.close()

###===========================================================================

