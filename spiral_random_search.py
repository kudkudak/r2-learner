import sys
import os
sys.path.append("..")

from sklearn import svm, datasets
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.base import BaseEstimator, clone 
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import os
import math
import numpy as np
import sklearn.metrics
from multiprocessing import Pool
from functools import partial

import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler

from data_api import *

from r2 import *
from lmao import *





def create_projector_2d_3(alpha_1, alpha_2, scale, b):
   return [ np.array([[np.cos(alpha_2/180.0*3.14)],[np.sin(alpha_2/180.0*3.14)]]).dot(np.array([[np.cos(alpha_1/180.0*3.14), np.cos(alpha_2/180.0*3.14)]])),
           b*np.array([[np.cos(alpha_2/180.0*3.14), np.sin(alpha_2/180.0*3.14)]]) ]
  
def create_random_projector_2d_3():
    return create_projector_2d_3(np.random.uniform(0,360), np.random.uniform(0,360), np.random.uniform(0,5), np.random.uniform(-1,1))


def test(p, X, Y):

    base_cls = partial(SVC, class_weight='auto', kernel='linear', C=1)
    model = LMAO(depth=3, base_cls=base_cls, projectors=p, \
                 scale=True, activation='sigmoid')
    # model = R2SVMLearner(despth=19, beta=0.1, use_prev=True, scale=True, activation='sigmoid')
    model.fit(X, Y)

    return sklearn.metrics.accuracy_score(Y, model.predict(X))



from multiprocessing import Value, Pool, Lock
import itertools

g_n = None
g_lock = None

def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

def initializer(n, lock):
	global g_n, g_lock
	g_n = n
	g_lock = lock

def f(inp):
	try:
		X, Y = np.loadtxt("data/two_spirals.x", skiprows=1), np.loadtxt("data/two_spirals.y", skiprows=1)
		i, p = inp[0] # Extract job data
		results = [test(p, X, Y) for i,p in inp] # Perform jobs
		j = np.argmax(results)
		a = results[j]
		with g_lock:
			if g_n.value < a:	
				g_n.value = a
				with open("spiral.out", "w") as g:		
					g.write(str(inp[j][1])+":"+str(a) + "\n")
		with open("spiral.counter.out", "w") as g:		
			g.write(str(i) + "\n")
	except KeyboardInterrupt:
		print "Thread interrupted by control-c"						

if __name__ == '__main__':
	num = Value('f', 0.0)

	with open("spiral.out", "w") as g:		
		g.write("0:0")

	with open("spiral.counter.out", "w") as g:		
		g.write("0")

	target = 1e8
	N = 100000
	chunk = 1000
	for j in range(int(target/N)):
		jobs = [(i+j*N, [create_random_projector_2d_3(), create_random_projector_2d_3()]) for i in xrange(N)]
		print "Running jobs ", float(N)*j/target
		p = Pool(3, initializer, [num, Lock()])
		print jobs[0:1], chunk
		r = p.map_async(f, grouper(chunk, jobs))
		try:
			r = r.get(target)
		except KeyboardInterrupt:
			print "Parent interrupted by control-c"
			p.terminate()
			sys.exit()
	print "Result ",num.value
