"""
Basic optimization through Tree of Parzen Estimators (often better than GP!)

How to use it? 

0. Download and install hyperopt, it is sufficient to do
    a) git clone https://github.com/hyperopt/hyperopt
    b) cd hyperopt && python setup.py install 

1. Define any function, for example

    def RF_logloss(max_depth, min_leaf_size):

   which returns a logloss of RF with given parameters

2. Define parameter space, in our case integer values

   space = { 'max_depth' : hp.choice('max_depth', range(10,100)),
             'min_leaf_size': hp.choice('min_leaf_size', range(1,5)) }

   while list is available at:
    https://github.com/hyperopt/hyperopt/wiki/FMin

3. Run optimize

   result = optimize(RF_logloss, space)

4. PROFIT!

### NOTE

    optimize MINIMIZES the function, so be carefull what your objective
    returns and use "-" if needed :-)



"""

import time
from hyperopt import fmin, tpe, hp, STATUS_OK
import datetime

def optimize(function, space, iters=100):

    def dump_and_call(function,log_file,kargs):
        value = function(**kargs)
        with open(log_file, 'a') as log:
            log.write(str(kargs)+':\t' + str(value) + '\n')
        return value
    
    def objective(function):
        log_file = str(datetime.datetime.now())+'.log'
        return lambda kargs:{'loss': dump_and_call(function,log_file,kargs), 'status': STATUS_OK }

    return fmin(objective(function),
        space=space,
        algo=tpe.suggest,
        max_evals=iters)

###############################

if __name__ == "__main__":

    def func(x,y):
        return x**2-y

    print 'Optimizing x^2-y over [-10,10], check out the DATE.log file!'
    print optimize(func, { 'x': hp.uniform('x', -10, 10), 'y':hp.uniform('y', -10, 10) })

