import gym
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from gym import error, spaces, utils
from gym.utils import seeding

class AllenCahn(gym.Env):
    metadata = {'render.modes': ['human']}

    class Config(object):
        n_layer = 4
        batch_size = 64
        valid_size = 256
        step_boundaries = [2000, 4000]
        num_iterations = 6000
        logging_frequency = 100
        verbose = True
        y_init_range = [0, 1]

    def __init__(self, dim=100, sigma=np.sqrt(2.0), num_time_interval, sqrt_delta_t, total_time):
        self._dim = dim

        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = sqrt_delta_t

        self._sigma = simga
        self._num_time_interval = num_time_interval
        self._total_time = total_time

        # initialize state variables to keep track of
        self.reset()

    def f_tf(self, t, x, y, z):
        return y - np.pow(y, 3)

    def g_tf(self, t, x):
        #return 0.5 / (1 + 0.2 * tf.reduce_sum(tf.square(x), 1, keep_dims=True))
        return 0.5 / (1 + 0.2 * np.sum(np.square(x), 1, keep_dims=True))

    def step(self, z): # z is action
        if self._step_counter > 0:
            self._x[:, :, self._step_counter] = self._x[:, :, self._step_counter-1] + self._sigma * self._dw_sample[:, :, self._step_counter]

            #self._y = self._y - self._deta_t * self.f_tf(self._time_stamp, self._x[:,:,self._step_counter], self._y, z) + tf.reduce_sum(z * self._dw[:, :, self._step_counter], 1, keep_dims=True)
            self._y = self._y - self._deta_t * self.f_tf(self._time_stamp, self._x[:,:,self._step_counter], self._y, z) + np.sum(z * self._dw[:, :, self._step_counter], 1, keep_dims=True)
        if self._time_stamp < self._total_time:
            reward = 0
            done = False
        else:
            reward = self._y - self.g_tf(self._total_time, self._x[:,:,-1])
            done = True

        return self._y, reward, done, {}

    def reset(self):
        self._x_init = np.zeros(self._dim)
        self._x = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
        self._x[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
        self._dw = normal.rvs(size=[num_sample,
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t

        self._y_init = tf.Variable(tf.random_uniform([1],
                                                     minval=self._config.y_init_range[0],
                                                     maxval=self._config.y_init_range[1],
                                                     dtype=TF_DTYPE))
        self._z_init = tf.Variable(tf.random_uniform([1, self._dim],
                                               minval=-.1, maxval=.1,
                                               dtype=TF_DTYPE))
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(self._dw)[0], 1]), dtype=TF_DTYPE)
        self._y = all_one_vec * self._y_init
        #self._z = tf.matmul(all_one_vec, z_init)
        self._step_counter = 0
        self._time_stamp = 0

    def render(self, mode='human'):
        ...
    def close(self):
        ...
