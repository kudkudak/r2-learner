import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from models import R2SVMLearner
import math

class Bunch(dict):
    """Container object for datasets: dictionary-like object that exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def fetch_uci_datasets(name=None):
    """Returns dict-like object (Bunch) contaning UCI datasets"""

    iris = datasets.load_iris()
    liver = datasets.fetch_mldata('liver-disorders')
    segment = datasets.fetch_mldata('segment')
    satimage = datasets.fetch_mldata('satimage')
    wine = datasets.fetch_mldata('uci-20070111 wine')

    iris.name = 'iris'
    liver.name = 'liver'
    segment.name = 'segment'
    satimage.name = 'satimage'
    wine.name = 'wine'

    heart_x, heart_y = datasets.load_svmlight_file('./data/heart')
    heart_x = heart_x.toarray()
    heart = Bunch(**{'name': 'heart', 'data': heart_x, 'target': heart_y, 'DESCR': 'libsvm heart data set'})

    uci_datasets = [iris, liver, segment, satimage, wine, heart]

    for dataset in uci_datasets :
        dataset.n_class = set(dataset.target)
        dataset.n_dim = dataset.data.shape[1]

    uci_datasets_bunch = Bunch(**{d.name: d for d in uci_datasets})
    if name is None :
        return uci_datasets_bunch
    else :
        return uci_datasets_bunch[name]


def fetch_synthetic_datasets():
    """Returns dict-like object (Bunch) contaning synthetic datasets"""

    X_moon, Y_moon = sklearn.datasets.make_moons(n_samples=1000, noise=0.04)
    X_spiral, Y_spiral = np.loadtxt("data/two_spirals.x", skiprows=1), np.loadtxt("data/two_spirals.y", skiprows=1)

    moon = Bunch(**{'name': 'moon', 'data': X_moon, 'target': Y_moon, 'DESCR': 'synthetic two moons data set', 'n_class': 2, 'n_dim': 2})
    spiral = Bunch(**{'name': 'spiral', 'data': X_spiral, 'target': Y_spiral, 'DESCR': 'synthetic two spirals data set', 'n_class': 2, 'n_dim': 2})

    return Bunch(**{'moon': moon, 'spiral': spiral})


# TODO: move this to different file?
def plot_contour(model, n=300, X=None, Y=None ):
    """Contour plot with option to provide original traning data"""

    x, y = np.linspace(X.min(), X.max(), n),  np.linspace(X.min(), X.max(), n)
    xx, yy = np.meshgrid(x, y)
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # z = model._feed_forward(np.c_[xx.ravel(), yy.ravel()])[:,0]
    plt.contourf(xx, yy, z.reshape(n,n), cmap=plt.cm.Paired, alpha=0.8)
    if X is not None and Y is not None :
        assert len(X) == len(Y)
        plt.scatter(X[:,0], X[:,1], c=Y,  cmap=plt.cm.Paired, alpha=0.8)
    plt.show()


# TODO: make both models compatible, move this to different file?
def run_and_plot(X, Y, **R2_params):
    plot_side = int(math.sqrt(R2_params['depth']))
    if plot_side**2 < R2_params['depth']: plot_side += 1

    for i in range(R2_params['depth']):
        model = R2SVMLearner(**R2_params)
        model.fit(X, Y)

        plt.subplot(plot_side, plot_side, i + 1)

        N = 200

        x, y = np.linspace(X.min(), X.max(), N),  np.linspace(X.min(), X.max(), N)
        xx, yy = np.meshgrid(x, y)
        z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        plt.contourf(xx, yy, z.reshape(N,N), cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:,0], X[:,1], c=Y,  cmap=plt.cm.Paired, alpha=0.8)
        plt.show()

