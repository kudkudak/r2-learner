import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import k_fold
from data_api import fetch_uci_datasets
from sklearn.svm import SVC
import time

n_jobs = 4

assert len(sys.argv) in [1,2]

if len(sys.argv) == 2:
    dataset =  sys.argv[1]
else :
    dataset = 'iris'

params = {'C': [np.exp(i) for i in xrange(-2,6)],
          'kernel': ['rbf'],
          'gamma': [np.exp(i) for i in xrange(-10,11)],
          'class_weight': ['auto']}

data = fetch_uci_datasets([dataset])[0]
model = SVC()
param_list = ParameterGrid(params)
exp_name = 'test'

def gen_params():
    for i, param in enumerate(param_list):
        yield {'model': model, 'params': param, 'data': data, 'name': exp_name}

params = list(gen_params())

def run(p):
    k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'])
#
# for p in params:
#     run(p)

p = Pool(n_jobs)
rs = p.map_async(run, params)
while True :
    if (rs.ready()): break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete on", data.name
    time.sleep(10)