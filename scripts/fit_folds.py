import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.grid_search import GridSearchCV, ParameterGrid
import numpy as np
from multiprocessing import Pool
from fit_models import k_fold
from data_api import fetch_uci_datasets
from r2 import R2SVMLearner

n_jobs = 16

assert len(sys.argv) in [1,2]

if len(sys.argv) == 2:
    dataset =  sys.argv[1]
else :
    dataset = 'glass'

params = {'C': [np.exp(i) for i in xrange(-2, 6)],
          'beta': [0.05 * i for i in xrange(1, 5)],
          'depth': [i for i in xrange(2,10,3)],
          'fit_c': ['random'],
          'activation': ['sigmoid'],
          'scale': [True, False], #[True, False],
          'recurrent': [True, False],
          'use_prev': [True, False],
          'seed': [666]}

data = fetch_uci_datasets([dataset])[0]

model = R2SVMLearner()


param_list = ParameterGrid(params)

def gen_params():
    for param in param_list:
        yield {'model': model, 'params': param, 'data': data}

params = list(gen_params())

def run(p):
    k_fold(base_model=p['model'], params=p['params'], data=p['data'])


p = Pool(n_jobs)
p.map(run, params)