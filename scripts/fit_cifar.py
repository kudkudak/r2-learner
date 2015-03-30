#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import traceback
from sklearn.grid_search import ParameterGrid
from multiprocessing import Pool
from fit_models import k_fold, extern_k_fold
from data_api import *
from r2 import R2SVMLearner
import time
from sklearn.svm import SVC, LinearSVC

n_jobs = 40

r2svm_params = {'beta': [0.1, 0.5, 1.0, 1.5, 2.0],
          'fit_c': ['random'],
          'scale': [True, False],
          'recurrent': [True, False],
          'use_prev': [True, False],
          'seed': [666]}

svm_params = {'C': [np.exp(i) for i in xrange(-7,7)],
              'gamma': [np.exp(i) for i in xrange(-10,11)]}

liner_svm_params = {'C': [np.exp(i) for i in xrange(-7,7)],
                    'loss':['l1']}

exp_params = [{'model': R2SVMLearner, 'params': r2svm_params, 'exp_name': 'test', 'model_name': 'r2svm'},
              {'model': SVC, 'params': svm_params, 'exp_name': 'test', 'model_name': 'svm'},
              {'model': LinearSVC, 'params': liner_svm_params, 'exp_name': 'test', 'model_name': 'linear_svm'}]


# cifar = fetch_cifar()
# print "X", cifar.data.shape
# print "Y", len(cifar.target)
# pca = PCA(n_components=0.9)
# pca.fit(cifar.data)
# cifar.data = pca.transform(cifar.data)
# print "Data shape after PCA:", cifar.data.shape
#
# f = open(os.path.join(c['DATA_DIR'], 'cifar.1'), 'w')
# cPickle.dump(cifar, f)

cifar = cPickle.load(open(os.path.join(c['DATA_DIR'], 'cifar.1'), 'r'))
cifar.target = np.array(cifar.target)
print cifar.data.shape

def gen_params():
    for r in exp_params:
        param_list = ParameterGrid(r['params'])
        for param in param_list:
            yield {'model': r['model'], 'params': param, 'data': cifar,
                   'name': r['exp_name'], 'model_name': r['model_name']}


params = list(gen_params())

def run(p):
    try:
        if p['model_name'] == 'r2svm':
            k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
                   model_name=p['model_name'], all_layers=True)
        else :
            extern_k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
                          model_name=p['model_name'])

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