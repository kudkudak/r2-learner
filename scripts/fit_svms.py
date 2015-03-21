import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import extern_k_fold
from data_api import *
from sklearn.svm import SVC
import time

n_jobs = 8

params = {'C': [np.exp(i) for i in xrange(-2,6)],
          'kernel': ['rbf'],
          'gamma': [np.exp(i) for i in xrange(-10,11)],
          'class_weight': ['auto']}

datasets = fetch_new_datasets()
datasets += fetch_small_datasets()

model = SVC()
param_list = ParameterGrid(params)
exp_name = 'test'

def gen_params():
    for data in datasets:
        for param in param_list:
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'svc'}

params = list(gen_params())

def run(p):
    extern_k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'], model_name=p['model_name'],
                  log=False, save_model=False)

# for p in params:
#     run(p)

p = Pool(n_jobs)
rs = p.map_async(run, params)
while True :
    if (rs.ready()): break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete for svc"
    time.sleep(10)