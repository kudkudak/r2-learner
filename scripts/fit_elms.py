import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import extern_k_fold
from data_api import *
from elm import ELM
import time

n_jobs = 8

params = {'h': [i for i in xrange(20, 101, 20)],
          'C': [10**i for i in xrange(0, 7)],
          'activation' : ['sigmoid'],
          'random_state': [666]}

datasets = fetch_new_datasets()
datasets += fetch_small_datasets()

model = ELM()
param_list = ParameterGrid(params)
exp_name = 'test'

def gen_params():
    for data in datasets:
        for param in param_list:
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'elm'}

params = list(gen_params())

def run(p):
    extern_k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'], model_name=p['model_name'])

# for p in params:
#     run(p)

p = Pool(n_jobs)
rs = p.map_async(run, params)
while True :
    if (rs.ready()): break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete for elm"
    time.sleep(10)