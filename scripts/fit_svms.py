#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sklearn.grid_search import GridSearchCV, ParameterGrid
from multiprocessing import Pool
from fit_models import extern_k_fold
from data_api import *
from sklearn.svm import SVC
import time
import traceback

n_jobs = 16

params = {'C': [np.exp(i) for i in xrange(-7,7)],
          'gamma': [np.exp(i) for i in xrange(-10,11)]}


datasets = fetch_uci_datasets(['vowel', 'vehicle'], tripled=True)

print " ".join([data.name for data in datasets])

model = SVC
param_list = ParameterGrid(params)
exp_name = 'triple'

def gen_params():
    for data in datasets:
        for param in param_list:
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'svm'}

params = list(gen_params())

def run(p):
    try:
        extern_k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
                      model_name=p['model_name'])
    except Exception:
            print p['model']
            print traceback.format_exc()

p = Pool(n_jobs)
rs = p.map_async(run, params, 1)
while True :
    if rs.ready():
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete for svc"
    time.sleep(3)