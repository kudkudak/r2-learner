import numpy as np
from sklearn.cross_validation import KFold

seed = 666

grid_config = {'experiment_type': 'grid',
               'refit': True,
               'scoring': 'accuracy',
               'store_clf': True,}

fold_config = {'experiment_type': 'k-fold',
               'n_folds': 5,
               'fold_seed': seed,
               'store_clf': True}

r2svm_params = {'C': [np.exp(i) for i in xrange(-2,6)],
                'beta': [0.04 * i for i in xrange(1,6)],
                'depth': [i for i in xrange(1,11)],
                'fit_c': [None],
                'scale': [True, False],
                'recurrent': [True, False],
                'use_prev': [True, False],
                'seed': [seed]}

r2elm_params = {'h': [i for i in xrange(20,201,45)],
                'beta': [0.04 * i for i in xrange(1,11)],
                'fit_h': [None], # 'grid', 'random' or None
                'depth': [i for i in xrange(1,11)],
                'scale': [True, False],
                'recurrent': [True, False],
                'use_prev': [True, False],
                'seed': [seed]}

elmrbf_params = {'h': [i for i in xrange(20, 201, 10)],
                 'activation' : ['rbf'],
                 'seed': [seed]}

elmsig_params = {'h': [i for i in xrange(20, 201, 10)],
                 'activation' : ['sigmoid'],
                 'seed': [seed]}

svmrbf_params = {'C': [np.exp(i) for i in xrange(-2,6)],
                 'kernel': ['rbf'],
                 'gamma': [np.exp(i) for i in xrange(-10,11,2)],
                 'class_weight': ['auto']}

r2svm_params_fit_c = {'C': [1],
                      'beta': [0.05 * i for i in xrange(0,5)],
                      'depth': [5,10],
                      'fit_c': ['random'],
                      'scale': [True],
                      'recurrent': [True],
                      'use_prev': [True, False],
                      'seed': [seed]}
