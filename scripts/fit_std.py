import sys, os
sys.path.append('..')

from misc.config import c
from data_api import *
import pandas as pd
from data_api import *
results_dir = c['RESULTS_DIR']
from multiprocessing import Pool
csv_results = {}
csv_dir = os.path.join(results_dir, 'csv')
import traceback
from fit_models import k_fold
import time

for csv_file in os.listdir(csv_dir):
    # print csv_file
    csv_results[csv_file] = pd.DataFrame.from_csv(os.path.join(csv_dir, csv_file))

models = ['test_r2elm', 'triple_r2svm']
datasets = ['glass', 'australian', 'bank','breast_cancer', 'crashes', 'liver', 'segment', 'satimage', 'heart', 'svmguide2'
            'diabetes', 'fourclass', 'german', 'indian', 'ionosphere', 'sonar', 'splice', 'iris', 'wine', 'pendigits',
            'svmguide4']

d = {model_name: { data_name: {} for data_name in datasets } for model_name in models}

for model in models:
    for data in datasets:
        if model + '_' + data in csv_results.keys():
            df = csv_results[model + '_' + data]
            d[model][data]['beta'] = df.loc[df['mean_acc'].idxmax(), 'beta']
            d[model][data]['fit_c'] = df.loc[df['mean_acc'].idxmax(), 'fit_c']
            d[model][data]['recurrent'] = df.loc[df['mean_acc'].idxmax(), 'recurrent']
            d[model][data]['scale'] = df.loc[df['mean_acc'].idxmax(), 'scale']
            d[model][data]['use_prev'] = df.loc[df['mean_acc'].idxmax(), 'use_prev']
            d[model][data]['seed'] = df.loc[df['mean_acc'].idxmax(), 'seed']
            if model == 'test_r2elm':
                d[model][data]['h'] = df.loc[df['mean_acc'].idxmax(), 'h']

for exp_name, datasets in d.iteritems():
    for params in datasets.iteritems():
        pass

n_jobs = 40

def run(p):
    try:
        k_fold(base_model=p['model'], params=p['params'], data=p['data'], exp_name=p['name'],
               model_name=p['model_name'], all_layers=True, save_model=True)
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