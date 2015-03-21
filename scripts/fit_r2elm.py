import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import k_fold
from data_api import fetch_new_datasets, fetch_small_datasets, fetch_uci_datasets
from r2 import R2ELMLearner
import time

n_jobs = 16

params = {'h': [i for i in xrange(20,101,20)],
          'beta': [0.1, 0.5, 1.0, 1.5, 2.0],
          'fit_c': ['random'],
          'scale': [True, False],
          'recurrent': [True, False],
          'use_prev': [True, False],
          'seed': [666]}

params = {'h': [10],
                    'beta': [0.1, 0.5],
                    'fit_c': ['random'],
                    'scale': [True],
                    'recurrent': [True],
                    'use_prev': [True,],
                    'seed': [666]}

#datasets = fetch_new_datasets()
# datasets = fetch_small_datasets()
datasets = fetch_uci_datasets(['heart'])

model = R2ELMLearner()
param_list = ParameterGrid(params)
exp_name = 'test'

def gen_params():
    for data in datasets:
        for i, param in enumerate(param_list):
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'r2elm'}

params = list(gen_params())

def run(p):
    k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'], model_name=p['model_name'],
           log=False, save_model=False)

# for p in params:
#     run(p)

pool = Pool(n_jobs)
rs = pool.map_async(run, params)
while True :
    if rs.ready():
        print "Ending", rs._number_left
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete for r2elm"
    time.sleep(1)