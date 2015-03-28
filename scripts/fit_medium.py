#!/usr/bin/env python

import sys, os
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sklearn.grid_search import ParameterGrid
from multiprocessing import Pool
from fit_models import k_fold
from r2 import R2ELMLearner, R2SVMLearner, R2LRLearner
import time
from data_api import *

datasets = fetch_medium_datasets()

n_jobs = 8

r2svm_params = {'beta': [0.1, 0.5, 1.0, 1.5, 2.0],
                'fit_c': ['random', None],
                'scale': [True, False],
                'recurrent': [True, False],
                'use_prev': [True, False],
                'seed': [666]}

r2elm_params = {'h': [i for i in xrange(20,101,20)],
                'beta': [0.1, 0.5, 1.0, 1.5, 2.0],
                'activation': ['sigmoid'],
                'fit_c': ['random', None],
                'scale': [True, False],
                'recurrent': [True, False],
                'use_prev': [True, False],
                'seed': [666]}

fixed_r2svm_params = {'beta': [0.1, 0.5, 1.0, 1.5, 2.0],
                     'depth': [i for i in xrange(1,11)],
                     'fit_c': ['random', None],
                     'scale': [True, False],
                     'recurrent': [True, False],
                     'use_prev': [True, False],
                     'seed': [666],
                     'fixed_prediction': [1]}

random_r2svm_params = {'beta': [0.1, 0.5, 1.0, 1.5, 2.0],
                       'depth': [i for i in xrange(1,11)],
                       'fit_c': ['random_cls'],
                       'scale': [True, False],
                       'recurrent': [True, False],
                       'use_prev': [True, False],
                       'seed': [666]}

exp_params = [{'model': R2SVMLearner, 'params': r2svm_params, 'exp_name': 'test', 'model_name': 'r2svm'},
              {'model': R2ELMLearner, 'params': r2elm_params, 'exp_name': 'test', 'model_name': 'r2elm'},
              {'model': R2SVMLearner, 'params': fixed_r2svm_params, 'exp_name': 'fixed', 'model_name': 'r2svm'},
              {'model': R2SVMLearner, 'params': random_r2svm_params, 'exp_name': 'random', 'model_name': 'r2svm'}]



def gen_params():
    for data in datasets:
        for r in exp_params:
            param_list = ParameterGrid(r['params'])
            for param in param_list:
                yield {'model': r['model'], 'params': param, 'data': data,
                       'name': r['exp_name'], 'model_name': r['model_name']}

params = list(gen_params())

def run(p):
    try:
        if p['exp_name'] == 'test':
            k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
                   model_name=p['model_name'], all_layers=True)
        else :
            k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
                   model_name=p['model_name'], all_layers=False)
    except:
        print p['model']
        print traceback.format_exc()

pool = Pool(n_jobs)
rs = pool.map_async(run, params, 1)

while True:
    if rs.ready():
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete"
    time.sleep(3)

