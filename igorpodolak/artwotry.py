__author__ = 'igor'

import sys
import numpy as np
import pylab as pl
from sklearn.datasets import make_moons, load_iris, load_digits
from sklearn.cross_validation import train_test_split

from utils.utils import get_data_bounds, plot_data, plot_contour, make_linearly_separable
from artwo.artwo import ArTwo


linear = (lambda x: x)
relu = (lambda x: np.maximum(0, x))
leakyrelu = (lambda x: np.maximum(x, 0.1 * x))
softplus = (lambda x: np.log(1.0 + np.exp(x)))
sigmoid = (lambda x: 1.0 / (1.0 + np.exp(-x)))
tanh = (lambda x: np.tanh(x))

act_fun_dict = {'linear': linear,
                'relu': relu,
                'leakyrelu': leakyrelu,
                'softplus': softplus,
                'sigmoid': sigmoid,
                'tanh': tanh}

def main():
    # construct dataset
    data_name = "moons(n_samples=1000, noise=0.15)"
    ds = make_moons(n_samples=1000, noise=0.15, random_state=0)
    # data_name = "separable"
    # ds = make_linearly_separable(n_samples=1000)
    X, y = ds
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,
                                                        random_state=0)
    # new classifier
    depth = 3
    hidden = 50
    alpha = 0.01
    act_fun_name = "linear"
    act_fun = act_fun_dict[act_fun_name]
    proj_fun_name = "linear"
    proj_fun = act_fun_dict[proj_fun_name]
    artwo = ArTwo(elm_act_fun=act_fun, proj_act_fun=proj_fun,
                  depth=depth, alpha=alpha, hidden=hidden)
    # fit the data
    score = artwo.fit(X=X_train, y=y_train)
    print "artwo.fit() --->", score

    # predict values
    out = artwo.predict(X=X_test)
    score = artwo.score(X=X_test, y=y_test)
    x_min, x_max, y_min, y_max, xx, yy = get_data_bounds(X=X)
    figure = pl.figure(figsize=(10, 5))
    ax = pl.subplot(1, 2, 1)
    plot_data(ax, X_train, y_train, X_test, y_test, xx, yy)
    Z = artwo.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax = pl.subplot(1, 2, 2)
    plot_contour(ax, X_train, y_train, X_test, y_test, xx, yy, Z, score)
    title = data_name + ", elm_fun = " + act_fun_name + \
            ", proj_fun = " + proj_fun_name + \
            ", depth = " + str(depth) + ", alpha = " + str(alpha)
    pl.suptitle(title)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                           wspace=0.01, hspace=0.01)
    pl.show()

from sklearn.cross_validation import cross_val_score, StratifiedKFold

def run_cv(dataset, act_fun_name="linear", proj_fun_name="linear",
           depth=10, hidden=50, alpha=0.05):
    act_fun = act_fun_dict[act_fun_name]
    proj_fun = act_fun_dict[proj_fun_name]
    # cross_val_score wymaga czegos od samego klasyfikatora...
    # scores = cross_val_score(artwo, dataset.data, dataset.target, cv=5)
    train_scores = []
    test_scores = []
    # split data
    skf = StratifiedKFold(dataset.target, n_folds=10)
    for train, test in skf:
        artwo = ArTwo(elm_act_fun=act_fun, proj_act_fun=proj_fun,
                      depth=depth, alpha=alpha, hidden=hidden)
        score = artwo.fit(X=dataset.data[train], y=dataset.target[train])
        train_scores.append(score[-1])
        score = artwo.score(X=dataset.data[test], y=dataset.target[test])
        test_scores.append(score)

    return train_scores, test_scores

if __name__ == '__main__':
    main()
    sys.exit()
    data = load_digits()
    for elm_fun in sorted(act_fun_dict.keys()):
        for proj_fun in sorted(act_fun_dict.keys()):
            print (elm_fun, proj_fun)
            train_mean = []
            test_mean = []
            for i in range(10):
                train_scores, test_scores = run_cv(data, act_fun_name=elm_fun, proj_fun_name=proj_fun)
                train_mean.append(np.mean(train_scores))
                test_mean.append(np.mean(test_scores))
            print "\ttrain", "%.4f" % np.mean(train_mean), "+-", "%.5f" % np.std(train_mean)
            print "\ttest ", "%.4f" % np.mean(test_mean), "+-", "%.5f" % np.std(test_mean)
