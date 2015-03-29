#!/usr/bin/env python

import sys, os
sys.path.append('..')
import cPickle
import pandas as pd
from misc.config import c
results_dir = c['RESULTS_DIR']

all_results = {}

datasets = ['segment', 'satimage', 'pendigits']
models = ['test_r2svm', 'test_r2elm', 'random_r2svm', 'fixed_r2svm']

paths = [ os.path.join(results_dir, model + '_' + dataset) for model in models for dataset in datasets ]

for path in paths:
    if os.path.isdir(path):
        print path
        results = {}
        for exp in os.listdir(path):
            name = exp[:-11]
            try:
                exp_res = cPickle.load(open(os.path.join(path, exp),'r'))
            except:
                print exp
                continue
            merged_res = exp_res['monitors']
            merged_res.update(exp_res['results'])
            merged_res.update(exp_res['config']['params'])
            results[name] = merged_res
        name = path.split('/')[-1]
        all_results[name] = results

for k, v in all_results.iteritems():
    pd.DataFrame.from_dict(v).transpose().to_csv(os.path.join(results_dir, 'csv', k))
