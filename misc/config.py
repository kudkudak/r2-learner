'''
Basic config for whole project. Modify those variables to reflect changes
'''
import os
import logging

base_dir = "/mnt/users/czarnecki/local/r2-learner"
name = "r2learner"

# Logger
def get_logger(name):
    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(name+': %(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    ch_file = logging.FileHandler(os.path.join(base_dir, name + ".log"))
    ch_file.setLevel(level = logging.INFO)
    ch_file.setFormatter(formatter)
    logger.addHandler(ch_file)
    return logger

# logger = get_logger("cores_r2learner")


# TODO: check what we need here
# Configurations
c = {
    "CACHE_DIR" : os.path.join(base_dir, "cache"),
    "DATA_DIR": os.path.join(base_dir, "data"),
    "RESULTS_DIR": os.path.join(base_dir, "results"),
    "BASE_DIR": base_dir,
    "LOG_DIR": os.path.join(base_dir, "logs"),
    "CURRENT_EXPERIMENT_CONFIG":{"experiment_name":"base_experiment_name"}
}
