#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import traceback
from sklearn.grid_search import GridSearchCV, ParameterGrid
from multiprocessing import Pool
from fit_models import k_fold
from data_api import *
from r2 import R2SVMLearner
import time

n_jobs = 16

params = {'beta': [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
          'fit_c': ['random_exhaustive'],
          'scale': [True, False],
          'recurrent': [True, False],
          'use_prev': [True, False],
          'seed': [666]}

datasets = fetch_uci_datasets(['fourclass', 'glass', 'ionosphere', 'sonar', 'vowel', 'splice', 'wine'])

print len(datasets)

model = R2SVMLearner
param_list = ParameterGrid(params)
exp_name = 'exh_'

def gen_params():
    for data in datasets:
        for i, param in enumerate(param_list):
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'r2svm'}

params = list(gen_params())

def run(p):
    try:
        k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
               model_name=p['model_name'], all_layers=True, save_model=True)
    except Exception:
        print p['params']
        print traceback.format_exc()

pool = Pool(n_jobs)
rs = pool.map_async(run, params, 1)

while True:
    if rs.ready():
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete"
    time.sleep(2)