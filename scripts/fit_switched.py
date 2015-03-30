#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sklearn.grid_search import ParameterGrid
from multiprocessing import Pool
from fit_models import k_fold
from data_api import *
from r2 import R2LRLearner
import time
import traceback

n_jobs = 1

params = {'depth': [i for i in xrange(1,11)],
          'fit_c': ['random', None],
          'scale': [True, False],
          'recurrent': [True, False],
          'use_prev': [True, False],
          'activation' : ['sigmoid'],
          'seed': [666],
          'switched': [True]}

datasets = fetch_new_datasets()
datasets += fetch_small_datasets()
datasets += fetch_medium_datasets()

model = R2LRLearner
param_list = ParameterGrid(params)
exp_name = 'switched'

def gen_params():
    for data in datasets:
        for i, param in enumerate(param_list):
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'r2lr'}

params = list(gen_params())

def run(p):
    try:
        k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],\
               model_name=p['model_name'], all_layers=False)
    except Exception:
            print p['data'].name
            print traceback.format_exc()

pool = Pool(n_jobs)
rs = pool.map_async(run, params, 1)
while True :
    if rs.ready():
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete"
    time.sleep(3)