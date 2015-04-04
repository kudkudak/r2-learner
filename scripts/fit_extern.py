#!/usr/bin/env python

# Fits ELM, Linear SVM and SVM RBF

import sys, os
from sklearn.grid_search import GridSearchCV, ParameterGrid
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import time
import traceback


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from misc.experiment_utils import save_exp, get_exp_logger, shorten_params, exp_done
from r2 import score_all_depths_r2, _r2_compress_model
from misc.data_api import *
from fit_models import *
from elm import ELM

n_jobs = 2

liner_svm_params = {'C': [np.exp(i) for i in xrange(-7,7)],
                    'loss': ['l1']}

svm_params = {'C': [np.exp(i) for i in xrange(-7,7)],
              'gamma': [np.exp(i) for i in xrange(-10,11)]}

elm_params = {'h': [i for i in xrange(20, 101, 20)],
              'C': [10**i for i in xrange(0, 7)],
              'activation': ['sigmoid'],
              'random_state': [666]}


datasets = fetch_all_datasets()


print " ".join([data.name for data in datasets])

exp_params = [{'model': LinearSVC, 'params': liner_svm_params, 'exp_name': 'test', 'model_name': 'linear_svm'},
              {'model': SVC, 'params': svm_params, 'exp_name': 'test', 'model_name': 'svm'},
              {'model': ELM, 'params': elm_params, 'exp_name': 'test', 'model_name': 'elm'}]


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
    print "Waiting for", remaining, "tasks to complete"
    time.sleep(3)
