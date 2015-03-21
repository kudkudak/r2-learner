import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import traceback
from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import k_fold
from data_api import *
from r2 import R2SVMLearner
import time
import logging
import multiprocessing as mlp


n_jobs = 8

assert len(sys.argv) in [1,2]

params = {'beta': [0.1],
          'fit_c': ['random_cls'],
          'activation': ['sigmoid'],
          'scale': [ False],
          'recurrent': [True],
          'use_prev': [True],
          'seed': [666]}

# params = {'beta': [0.1, 0.2, 0.3],
#           'fit_c': ['random_cls'],              # SINGLE
#           'activation': ['sigmoid'],        # SINGLE
#           'scale': [True],
#           'recurrent': [True],
#           'use_prev': [True],
#           'seed': [666]}

datasets = fetch_uci_datasets(['heart'])
model = R2SVMLearner()
param_list = ParameterGrid(params)
exp_name = 'test'

def gen_params():
    for data in datasets:
        for i, param in enumerate(param_list):
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'r2svm'}

params = list(gen_params())

def run(p):
    try:
        k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'], model_name=p['model_name'],
           save_model=False, log=False)
    except Exception:
        print p['params']
        print traceback.format_exc()

# for i, p in enumerate(params):
#     run(p)
#     print "done %i/%i on %s with %s" % (i+1, len(params), p['data'].name, p['model_name'])

pool = Pool(n_jobs)
rs = pool.map_async(run, params, 1)

while True:
    if rs.ready():
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete"
    time.sleep(2)