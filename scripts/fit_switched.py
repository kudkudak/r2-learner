import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import k_fold
from data_api import fetch_new_datasets, fetch_small_datasets, fetch_uci_datasets, fetch_medium_datasets
from r2 import R2ELMLearner
import time
import traceback

n_jobs = 3

params = {'h': [i for i in xrange(20,101,20)],
          'beta': [0.1, 0.5, 1.0, 1.5, 2.0],
          'activation': ['01_rbf'],
          'fit_c': ['random'],
          'scale': [True, False],
          'recurrent': [True, False],
          'use_prev': [True, False],
          'seed': [666],
          'switched': [True]}

datasets = fetch_new_datasets()
datasets += fetch_small_datasets()
# datasets += fetch_medium_datasets()



model = R2ELMLearner()
param_list = ParameterGrid(params)
exp_name = 'switched'

def gen_params():
    for data in datasets:
        for i, param in enumerate(param_list):
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'rbf_r2elm'}

params = list(gen_params())

def run(p):
    try:
        exp = k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],\
               model_name=p['model_name'])
        # print exp['results']
    except Exception:
            print p['model']
            print traceback.format_exc()

# for p in params:
#     run(p)

pool = Pool(n_jobs)
rs = pool.map_async(run, params, 1)
while True :
    if rs.ready():
        print "Ending", rs._number_left
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete for r2elm"
    time.sleep(1)