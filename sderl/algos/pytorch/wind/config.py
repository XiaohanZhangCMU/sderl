import numpy as np


class Config(object):
    n_layer = 4
    batch_size = 512
    valid_size = 256
    patience = 50
    step_boundaries = [2000, 4000]
    num_pts = 100
    logging_frequency = 1
    verbose = True
    y_init_range = [0, 1]


class AllenCahnConfig(Config):
    total_time = 0.3
    num_time_interval = 20
    train_iters=1500
    dim = 100
    lr_values = list(np.array([5e-4, 5e-4]))
    lr_boundaries = [2000]
    num_hiddens = [dim, dim + 10, dim + 10, dim]
    y_init_range = [0.3, 0.6]


class HJBConfig(Config):
    # Y_0 is about 4.5901.
    dim = 100
    total_time = 1.0
    num_time_interval = 3
    train_iters=1500
    lr_boundaries = [400]
    lr_values = list(np.array([1e-2, 1e-2]))
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [0, 1]


class PricingOptionConfig(Config):
    dim = 100
    total_time = 0.5
    num_time_interval = 20
    lr_values = list(np.array([5e-3, 5e-3]))
    lr_boundaries = [2000]
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [15, 18]


class PricingDefaultRiskConfig(Config):
    dim = 100
    total_time = 1
    num_time_interval = 40
    lr_values = list(np.array([8e-3, 8e-3]))
    lr_boundaries = [3000]
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [40, 50]


class BurgesTypeConfig(Config):
    dim = 50
    total_time = 0.2
    num_time_interval = 30
    train_iters=15000
    lr_values = list(np.array([1e-2, 1e-3, 1e-4]))
    lr_boundaries = [15000, 25000]
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [2, 4]


class QuadraticGradientsConfig(Config):
    dim = 100
    total_time = 1.0
    num_time_interval = 30
    train_iters=15000
    lr_values = list(np.array([1e-2, 5e-3]))
    lr_boundaries = [2000]
    num_hiddens = [dim, dim+10, dim+10, dim]
    y_init_range = [2, 4]


class ReactionDiffusionConfig(Config):
    dim = 100
    total_time = 1.0
    num_time_interval = 30
    train_iters=15000
    lr_values = list(np.array([1e-2, 1e-3, 1e-4]))
    lr_boundaries = [8000, 16000]
    num_hiddens = [dim, dim+10, dim+10, dim]


def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")
