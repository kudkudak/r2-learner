from multiprocessing import Pool
import numpy as np
import time

params = [x for x in xrange(10000)]
def run(x):
    x_sqrt = np.sqrt(x)
    # print x, x_sqrt
p = Pool(2)
rs = p.map_async(run, params)

while True :
    if (rs.ready()): break
    remaining = rs._job

    print "Waiting for", remaining, "tasks"
    time.sleep(0.2)