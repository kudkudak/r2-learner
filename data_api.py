import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.sparse import vstack
import math
import os
from misc.config import c
from sklearn.decomposition import PCA

data_dir = c["DATA_DIR"]

class Bunch(dict):
    """Container object for datasets: dictionary-like object that exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def fetch_uci_datasets(names, tripled=False):
    """Returns dict-like object (Bunch) contaning UCI datasets"""

    assert type(names) == list
    assert(os.path.exists(os.path.join(data_dir, 'heart')))
    assert(os.path.exists(os.path.join(data_dir, 'glass.scale')))
    assert(os.path.exists(os.path.join(data_dir, 'aloi.scale')))
    assert(os.path.exists(os.path.join(data_dir, 'pendigits')))
    assert(os.path.exists(os.path.join(data_dir, 'pendigits.t')))
    assert(os.path.exists(os.path.join(data_dir, 'news20.scale')))
    assert(os.path.exists(os.path.join(data_dir, 'news20.t.scale')))
    assert(os.path.exists(os.path.join(data_dir, 'covtype.scale')))
    assert(os.path.exists(os.path.join(data_dir, 'mnist.pkl')))
    # new
    assert(os.path.exists(os.path.join(data_dir, 'australian')))
    assert(os.path.exists(os.path.join(data_dir, 'bank')))
    assert(os.path.exists(os.path.join(data_dir, 'breast-cancer')))
    assert(os.path.exists(os.path.join(data_dir, 'crashes')))
    assert(os.path.exists(os.path.join(data_dir, 'diabetes')))
    assert(os.path.exists(os.path.join(data_dir, 'fourclass')))
    assert(os.path.exists(os.path.join(data_dir, 'german.numer')))
    assert(os.path.exists(os.path.join(data_dir, 'indian')))
    assert(os.path.exists(os.path.join(data_dir, 'ionosphere_scale')))
    assert(os.path.exists(os.path.join(data_dir, 'mushrooms')))
    assert(os.path.exists(os.path.join(data_dir, 'sonar_scale')))
    assert(os.path.exists(os.path.join(data_dir, 'splice')))
    assert(os.path.exists(os.path.join(data_dir, 'vowel.scale')))
    assert(os.path.exists(os.path.join(data_dir, 'vowel.scale.t')))
    assert(os.path.exists(os.path.join(data_dir, 'vehicle.scale')))
    assert(os.path.exists(os.path.join(data_dir, 'svmguide2')))
    assert(os.path.exists(os.path.join(data_dir, 'svmguide4')))
    assert(os.path.exists(os.path.join(data_dir, 'svmguide4.t')))

    uci_datasets = []

    if 'fourclass' in names:
        fourclass_x, fourclass_y = datasets.load_svmlight_file(os.path.join(data_dir, 'fourclass'))
        fourclass = Bunch(**{'name': 'fourclass', 'data': fourclass_x.toarray(), 'target': fourclass_y})
        uci_datasets.append(fourclass)
    if 'bank' in names:
        bank_x, bank_y = datasets.load_svmlight_file(os.path.join(data_dir, 'bank'))
        bank = Bunch(**{'name': 'bank', 'data': bank_x.toarray(), 'target': bank_y})
        uci_datasets.append(bank)
    if 'iris' in names:
        iris = datasets.load_iris()
        iris.name = 'iris'
        uci_datasets.append(iris)
    if 'wine' in names:
        wine = datasets.fetch_mldata('uci-20070111 wine')
        wine.name = 'wine'
        uci_datasets.append(wine)
    if 'liver' in names:
        liver = datasets.fetch_mldata('liver-disorders')
        liver.name = 'liver'
        uci_datasets.append(liver)
    if 'diabetes' in names:
        diabetes_x, diabetes_y = datasets.load_svmlight_file(os.path.join(data_dir, 'diabetes'))
        diabetes = Bunch(**{'name': 'diabetes', 'data': diabetes_x.toarray(), 'target': diabetes_y})
        uci_datasets.append(diabetes)
    if 'glass' in names:
        glass_x, glass_y = datasets.load_svmlight_file(os.path.join(data_dir, 'glass.scale'))
        glass = Bunch(**{'name': 'glass', 'data': glass_x.toarray(), 'target': glass_y, 'DESCR': 'libsvm glass.scale'})
        uci_datasets.append(glass)
    if 'breast_cancer' in names:
        breast_cancer_x, breast_cancer_y = datasets.load_svmlight_file(os.path.join(data_dir, 'breast-cancer'))
        breast_cancer = Bunch(**{'name': 'breast_cancer', 'data': breast_cancer_x.toarray(), 'target': breast_cancer_y})
        uci_datasets.append(breast_cancer)
    if 'indian' in names:
        indian_x, indian_y = datasets.load_svmlight_file(os.path.join(data_dir, 'indian'))
        indian = Bunch(**{'name': 'indian', 'data': indian_x.toarray(), 'target': indian_y})
        uci_datasets.append(indian)
    if 'vowel' in names:
        vowel_train_x, \
        vowel_train_y, \
        vowel_test_x, \
        vowel_test_y = datasets.load_svmlight_files((os.path.join(data_dir, 'vowel.scale'),
                                                   os.path.join(data_dir, 'vowel.scale.t')))
        vowel_x = vstack([vowel_train_x, vowel_test_x])
        vowel_y = np.hstack([vowel_train_y, vowel_test_y])
        vowel = Bunch(**{'name': 'vowel', 'data': vowel_x.toarray(), 'target': vowel_y, 'DESC': 'libsvm vowel dataset'})
        uci_datasets.append(vowel)
    if 'svmguide4' in names:
        svmguide4_train_x, \
        svmguide4_train_y, \
        svmguide4_test_x, \
        svmguide4_test_y = datasets.load_svmlight_files((os.path.join(data_dir, 'svmguide4'),
                                                   os.path.join(data_dir, 'svmguide4.t')))
        svmguide4_x = vstack([svmguide4_train_x, svmguide4_test_x])
        svmguide4_y = np.hstack([svmguide4_train_y, svmguide4_test_y])
        svmguide4 = Bunch(**{'name': 'svmguide4', 'data': svmguide4_x.toarray(), 'target': svmguide4_y, 'DESC': 'libsvm svmguide4 dataset'})
        uci_datasets.append(svmguide4)
    if 'heart' in names:
        heart_x, heart_y = datasets.load_svmlight_file(os.path.join(data_dir, 'heart'))
        heart_x = heart_x.toarray()
        heart = Bunch(**{'name': 'heart', 'data': heart_x, 'target': heart_y, 'DESCR': 'libsvm heart data set'})
        uci_datasets.append(heart)
    if 'australian' in names:
        australian_x, australian_y = datasets.load_svmlight_file(os.path.join(data_dir, 'australian'))
        australian = Bunch(**{'name': 'australian', 'data': australian_x.toarray(), 'target': australian_y})
        uci_datasets.append(australian)
    if 'pendigits' in names:
        pen_train_x, \
        pen_train_y, \
        pen_test_x, \
        pen_test_y = datasets.load_svmlight_files((os.path.join(data_dir, 'pendigits'),
                                                   os.path.join(data_dir, 'pendigits.t')))
        pen_x = vstack([pen_train_x, pen_test_x])
        pen_y = np.hstack([pen_train_y, pen_test_y])
        pendigits = Bunch(**{'name': 'pendigits', 'data': pen_x.toarray(), 'target': pen_y, 'DESC': 'libsvm pendigits dataset'})
        uci_datasets.append(pendigits)
    if 'vehicle' in names:
        vehicle_x, vehicle_y = datasets.load_svmlight_file(os.path.join(data_dir, 'vehicle.scale'))
        vehicle = Bunch(**{'name': 'vehicle', 'data': vehicle_x.toarray(), 'target': vehicle_y})
        uci_datasets.append(vehicle)
    if 'segment' in names:
        segment = datasets.fetch_mldata('segment')
        segment.name = 'segment'
        uci_datasets.append(segment)
    if 'svmguide2' in names:
        svmguide2_x, svmguide2_y = datasets.load_svmlight_file(os.path.join(data_dir, 'svmguide2'))
        svmguide2 = Bunch(**{'name': 'svmguide2', 'data': svmguide2_x.toarray(), 'target': svmguide2_y})
        uci_datasets.append(svmguide2)
    if 'crashes' in names:
        crashes_x, crashes_y = datasets.load_svmlight_file(os.path.join(data_dir, 'crashes'))
        crashes = Bunch(**{'name': 'crashes', 'data': crashes_x.toarray(), 'target': crashes_y})
        uci_datasets.append(crashes)
    if 'german' in names:
        german_x, german_y = datasets.load_svmlight_file(os.path.join(data_dir, 'german.numer'))
        german = Bunch(**{'name': 'german', 'data': german_x.toarray(), 'target': german_y})
        uci_datasets.append(german)
    if 'ionosphere' in names:
        ionosphere_x, ionosphere_y = datasets.load_svmlight_file(os.path.join(data_dir, 'ionosphere_scale'))
        ionosphere = Bunch(**{'name': 'ionosphere', 'data': ionosphere_x.toarray(), 'target': ionosphere_y})
        uci_datasets.append(ionosphere)
    if 'satimage' in names:
        satimage = datasets.fetch_mldata('satimage')
        satimage.name = 'satimage'
        uci_datasets.append(satimage)
    if 'sonar' in names:
        sonar_x, sonar_y = datasets.load_svmlight_file(os.path.join(data_dir, 'sonar_scale'))
        sonar = Bunch(**{'name': 'sonar', 'data': sonar_x.toarray(), 'target': sonar_y})
        uci_datasets.append(sonar)
    if 'splice' in names:
        splice_x, splice_y = datasets.load_svmlight_file(os.path.join(data_dir, 'splice'))
        splice = Bunch(**{'name': 'splice', 'data': splice_x.toarray(), 'target': splice_y})
        uci_datasets.append(splice)
    if 'mushrooms' in names:
        mushrooms_x, mushrooms_y = datasets.load_svmlight_file(os.path.join(data_dir, 'mushrooms'))
        mushrooms = Bunch(**{'name': 'mushrooms', 'data': mushrooms_x.toarray(), 'target': mushrooms_y})
        uci_datasets.append(mushrooms)


    if 'aloi' in names:
        aloi_x, aloi_y = datasets.load_svmlight_file(os.path.join(data_dir, 'aloi.scale'))
        aloi = Bunch(**{'name': 'aloi', 'data': aloi_x, 'target': aloi_y, 'DESCR':'libsvm aloi.scale data set'})
        uci_datasets.append(aloi)
    if 'news20' in names:
        news_train_x, \
        news_train_y, \
        news_test_x, \
        news_test_y = datasets.load_svmlight_files((os.path.join(data_dir, 'news20.scale'),
                                                    os.path.join(data_dir, 'news20.t.scale')))
        news_x = vstack([news_train_x, news_test_x])
        news_y = np.hstack([news_train_y, news_test_y])
        news20 = Bunch(**{'name': 'news20', 'data': news_x, 'target': news_y, 'DESCR': 'libsvm scaled 20 news groups'})
        uci_datasets.append(news20)
    if 'covtype' in names:
        covtype_x, covtype_y = datasets.load_svmlight_file(os.path.join(data_dir, 'covtype.scale'))
        covtype = Bunch(**{'name': 'covtype', 'data': covtype_x, 'target': covtype_y, 'DESCR': 'libsvm covtype'})
        uci_datasets.append(covtype)
    if 'mnist' in names:
        f = open(os.path.join(data_dir, 'mnist.pkl'))
        train_set, valid_set, test_set = pickle.load(f)
        f.close()
        mnist_x = np.vstack([train_set[0], valid_set[0], test_set[0]])
        mnist_y = np.hstack([train_set[1], valid_set[1], test_set[1]])
        mnist = Bunch(**{'name': 'mnist', 'data': mnist_x, 'target': mnist_y})
        uci_datasets.append(mnist)

    for dataset in uci_datasets :
        if tripled:
            dataset.data = np.hstack([dataset.data for _ in range(3)])
        dataset.n_class = len(set(dataset.target))
        dataset.n_dim = dataset.data.shape[1]
        dataset.n_rows = dataset.data.shape[0]
        dataset.man_size = manifold_dim(dataset.data)

    return uci_datasets


def fetch_tripled_datasets():
    return fetch_uci_datasets(['fourclass', 'iris', 'wine', 'liver', 'diabetes', 'bank', 'glass', 'breast_cancer', 'indian',
                               'heart', 'australian', 'crashes', 'german', 'ionosphere', 'sonar', 'splice'],
                              tripled=True)

def fetch_medium_tripled_datasets():
    return fetch_uci_datasets(['satimage', 'segment', 'pendigits'], tripled=True)

def fetch_small_datasets():
    return fetch_uci_datasets(['iris', 'liver', 'heart', 'wine', 'glass'])


def fetch_binray_datasets():
    return fetch_uci_datasets(['liver', 'heart'])


def fetch_large_datasets():
    return fetch_uci_datasets(['mnist', 'news20', 'covtype', 'aloi'])


def fetch_medium_datasets():
    return fetch_uci_datasets(['segment', 'satimage', 'pendigits'])


def fetch_all_datasets(): # for when shit gets real
    return fetch_uci_datasets(['iris', 'liver', 'heart', 'wine', 'glass', 'segment', 'satimage', \
                               'pendigits', 'mnist', 'news20', 'covtype', 'aloi'])

def fetch_new_datasets():
    return fetch_uci_datasets(['australian', 'bank', 'breast_cancer', 'crashes', 'diabetes', 'fourclass', \
                               'german', 'indian', 'ionosphere', 'mushrooms', 'sonar', 'splice', 'vehicle', 'vowel'])


def fetch_synthetic_datasets():
    """Returns dict-like object (Bunch) contaning synthetic datasets"""

    X_moon, Y_moon = sklearn.datasets.make_moons(n_samples=1000, noise=0.04)
    X_spiral, Y_spiral = np.loadtxt(os.path.join(c["DATA_DIR"], "two_spirals.x"), skiprows=1), \
                         np.loadtxt(os.path.join(c["DATA_DIR"], "two_spirals.y"), skiprows=1)

    moon = Bunch(**{'name': 'moon', 'data': X_moon, 'target': Y_moon,
                    'DESCR': 'synthetic two moons data set', 'n_class': 2, 'n_dim': 2})
    spiral = Bunch(**{'name': 'spiral', 'data': X_spiral, 'target': Y_spiral,
                      'DESCR': 'synthetic two spirals data set', 'n_class': 2, 'n_dim': 2})

    return Bunch(**{'moon': moon, 'spiral': spiral})


def shuffle_data(data, seed=None) :
    X, Y = data.data, data.target
    assert X.shape[0] == Y.shape[0]

    if seed is not None:
        np.random.seed(seed)
    p = np.random.permutation(len(X))

    data.data = X[p]    # Copying data here!
    data.target = Y[p]

    return data

def manifold_dim(X, thresh=0.95):
    pca = PCA()
    pca.fit(X)
    expl = 0
    for v, var in enumerate(pca.explained_variance_ratio_):
        expl += var
        if expl >= thresh:
            break
    return v+1

