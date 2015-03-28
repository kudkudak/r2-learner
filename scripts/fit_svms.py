import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import extern_k_fold
from data_api import *
from sklearn.svm import SVC
import time
import traceback

n_jobs = 1

params = {'C': [np.exp(i) for i in xrange(-7,7)],
          'kernel': ['rbf'],
          'gamma': [np.exp(i) for i in xrange(-10,11)]}

params = {'C': [1],
          'kernel': ['rbf'],
          'gamma': [0]}



# datasets = fetch_new_datasets()
# datasets += fetch_small_datasets()
# datasets += fetch_medium_datasets()

datasets = fetch_uci_datasets(['fourclass'])

print " ".join([data.name for data in datasets])

model = SVC()
param_list = ParameterGrid(params)
exp_name = 'unit_test'

def gen_params():
    for data in datasets:
        for param in param_list:
            yield {'model': model, 'params': param, 'data': data, 'name': exp_name, 'model_name': 'svc'}

params = list(gen_params())
print len(params)

def run(p):
    try:
        extern_k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
                  model_name=p['model_name'], save_model=True)
    except Exception:
            print p['model']
            print traceback.format_exc()

# for i, p in enumerate(params):
#     run(p)
#     print "done %i/%i with %s" % (i, len(params), p['model_name'])

p = Pool(n_jobs)
rs = p.map_async(run, params, 1)
while True :
    if rs.ready():
        break
    remaining = rs._number_left
    print "Waiting for", remaining, "tasks to complete for svc"
    time.sleep(3)