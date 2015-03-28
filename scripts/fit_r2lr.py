import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import k_fold
from data_api import *
from r2 import R2LRLearner
import time
import traceback

n_jobs = 8

params = { 'scale': [True, False],
          'recurrent': [True, False],
          'use_prev': [True, False],
          'activation' : ['sigmoid'],
          'seed': [666]}

datasets = fetch_new_datasets()
datasets += fetch_small_datasets()
datasets += fetch_medium_datasets()

model = R2LRLearner()
param_list = ParameterGrid(params)
exp_name = 'test'

def gen_params():
    for data in datasets:
        for param in param_list:
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'r2lr'}

params = list(gen_params())

def run(p):
    try:
        k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'], model_name=p['model_name'],
                  log=False)
    except:
        print p['model']
        print traceback.format_exc()

# for p in params:
#     run(p)

p = Pool(n_jobs)
rs = p.map_async(run, params)
while True :
    if (rs.ready()): break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete for lr"
    time.sleep(10)