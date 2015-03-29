from optparse import OptionParser
import cPickle
import logging
import os
from config import c

def get_logger(name, to_file=False):
    logger = logging.Logger(name=name, level=logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s \t" + name + "  %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if to_file:
        logger.propagate = False
        ch_file = logging.FileHandler(os.path.join(c["LOG_DIR"],name + ".log"))
        ch_file.setLevel(level=logging.INFO)
        ch_file.setFormatter(formatter)
        logger.addHandler(ch_file)
    return logger

def get_exp_logger(config, dir_name, to_file=False, to_std=True):
    assert to_file != to_std
    name = config["experiment_name"]
    logger = logging.Logger(name=name, level=logging.INFO)
    formatter = logging.Formatter("%(asctime)s \t" + name + "  %(message)s")
    if to_std:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if to_file:
        directory = os.path.join(c["LOG_DIR"], dir_name)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        logger.propagate = False
        ch_file = logging.FileHandler(os.path.join(directory, name + ".log"))
        ch_file.setLevel(level=logging.INFO)
        ch_file.setFormatter(formatter)
        logger.addHandler(ch_file)
    return logger


def get_exp_options(config):
    config["experiment_name"] = "my_experiment"

    parser = OptionParser()
    parser.add_option("-e", "--e_name", dest="experiment_name", default="my_experiment")

    for cn, cv in config.iteritems():
        if type(cv) == type(1.0) or type(cv) == type(1) or type(cv) == type(""):
            parser.add_option("", "--" + cn, dest=cn, default=cv, type=type(cv))

    (options, args) = parser.parse_args()

    return {cn: getattr(options, cn) for cn in config.iterkeys()}


def print_exp_name(config):
    print "Experiment " + config["experiment_name"]


def print_exp_header(config):
    print "Experiment " + config["experiment_name"]
    print "======================="
    for cn, cv in config.iteritems():
        print "\t " + cn + " = " + str(cv)
    print "\n"


def generate_configs(c, grid):
    counts = [len(c[key]) for key in grid]
    id = [0] * len(grid)

    while id[0] != counts[0]:
        new_c = dict(c)
        for i, g in enumerate(grid):
            new_c[g] = c[g][id[i]]

        new_c["experiment_name"] += "_".join([grid[i] + "=" + str(new_c[grid[i]]) for i in range(len(grid))])
        yield new_c

        # Iterate index
        id[-1] += 1
        for i in range(len(grid) - 1, 0, -1):
            if id[i] == counts[i]:
                id[i - 1] += 1
                id[i] = 0
            else:
                break


def get_exp_fname(E):
    return E["config"]["experiment_name"] + ".experiment"


def exp_done(E, dir_name):
    directory = os.path.join(c["RESULTS_DIR"], dir_name)
    path = os.path.join(directory, get_exp_fname(E))
    return os.path.exists(path)


def save_exp(E, dir_name):
    directory = os.path.join(c["RESULTS_DIR"], dir_name)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    cPickle.dump(E, open(os.path.join(directory, get_exp_fname(E)), "w"))

def shorten_params(params):
    short_params = ""
    for k, v in params.iteritems():
        if k in ['C', 'beta', 'h', 'scale', 'recurrent', 'use_prev', 'gamma', 'depth', 'fit_c']:
            short_params += str(k)[0]
            if type(v) == float:
                short_params += str(v)
            elif type(v) == int:
                short_params += str(v)
            elif type(v) == bool:
                if v: short_params += 'T'
                else: short_params += 'F'
            elif v is None:
                short_params += 'No'
            else:
                short_params += str(v)
            short_params += '_'

    return short_params
