import os
import re
import pandas as pd
import numpy as np

log_dir = '../scripts'

def process_model_name(name) :
    if name[:5] == 'R2SVM':
        return name[:5]
    elif name[:5] == 'R2ELM':
        return name[:5]
    elif name == "ELM+RBF":
        return 'ELM+RBF'
    elif name == "ELM+SIG":
        return 'ELM+SIG'
    elif name[:3] == "SVC":
        return 'SVM'
    else:
        print "bad name: ", name

models = ['R2SVM', 'R2ELM', 'ELM+RBF', 'ELM+SIG', 'SVM']
datasets = ['iris', 'liver', 'heart'] #, 'satimage', 'segment']
acc = {m: {d: 0. for d in datasets} for m in models}

for log in os.listdir(log_dir):
    if 'fold' in log and log[-3:] == 'log':
        model = process_model_name(log[:7])
        f = open(os.path.join(log_dir, log), 'r')
        s = f.read()
        score = re.search("'mean_acc': (0\.\d+)", s).group(1)
        name = re.search("'data_name': '(\w+)'", s).group(1)
        acc[model][name] = score

pd.options.display.float_format = '{:,.4f}'.format
acc = pd.DataFrame.from_dict(acc, dtype=np.float64)

print acc

