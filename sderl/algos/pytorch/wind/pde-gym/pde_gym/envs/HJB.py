import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Equation(gym.Env):
    """Base class for defining PDE related function."""
    metadata = {'render.modes': ['human']}

    def __init__(self, dim, total_time, num_time_interval):
        super(HJB, self).__init__()
        self._dim = dim
        self._total_time = total_time
        self._num_time_interval = num_time_interval
        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None
        self.ti = 0

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    def reset(self):
        ...
    def render(self, mode='human'):
        ...
    def close(self):
        ...

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


class HJB(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(HJB, self).__init__(dim, total_time, num_time_interval, num_sample)
        self._x_init = np.zeros(self._dim)
        self._sigma = np.sqrt(2.0)
        self._lambda = 50
        self._num_sample = num_sample

    def step(self, action):
        def sample():
            dw_sample = normal.rvs(size=[self._num_sample,
                                         self._dim,
                                         self._num_time_interval]) * self._sqrt_delta_t
            x_sample = np.zeros([self._num_sample, self._dim, self._num_time_interval + 1])
            x_sample[:, :, 0] = np.ones([self._num_sample, self._dim]) * self._x_init
            for i in range(self._num_time_interval):
                x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
            return torch.as_tensor(dw_sample, dtype=torch.float64), torch.as_tensor(x_sample, dtype=torch.float64)

        dw, x = sample()

        z = action[0]
        if t == 0:
            self.y = action[1]
            all_one_vec = torch.ones([dw.shape[0],1], dtype=torch.float64)
            self.y = all_one_vec * self.y

        self.y = self.y - self.delta_t * (
            self.f_tf(time_stamp[t], x[:, :, t], self.y, z)
        ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)

        if self.done:
            self.y = self.y - self.delta_t * self.f_tf(
                time_stamp[-1], x[:, :, -2], y, z
            ) + torch.sum(z * dw[:, :, -1], 1, keepdim=True)
            delta = y - self.g_tf(self._total_time, x[:, :, -1])
            rwd = np.abs(delta)
        else:
            rwd = 0

        return x, rwd, self.done, self.y

    def f_tf(self, t, x, y, z):
        tmp = torch.pow(z,2)
        return -self._lambda * torch.sum(tmp, 1, keepdim=True, dtype=torch.float64)

    def g_tf(self, t, x):
        return torch.log((1 + torch.sum(torch.pow(x,2), 1, keepdim=True, dtype=torch.float64)) / 2)





#class HJB(Equation):
#    def __init__(self):
#        super(HJB, self).__init__()
#
#        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
#        self.observation_space = spaces.Box(low=0, high=255,
#                                        shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
#
#        self._dim = dim
#        self._total_time = total_time
#        self._num_time_interval = num_time_interval
#        self._delta_t = (self._total_time + 0.0) / self._num_time_interval
#        self._sqrt_delta_t = np.sqrt(self._delta_t)
#        self._y_init = None
#
#        self._x_init = np.zeros(self._dim)
#        self._sigma = np.sqrt(2.0)
#        self._lambda = 1.0
#        self.done = True
#        self.num_time_interval = 20
#
#
#    def step(self, action):
#
#        act1 = action[0]
#        act2 = action[1]
#
#        def sample(num_sample):
#            dw_sample = normal.rvs(size=[num_sample,
#                                         self._dim,
#                                         self._num_time_interval]) * self._sqrt_delta_t
#            x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1])
#            x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init
#            for i in range(self._num_time_interval):
#                x_sample[:, :, i + 1] = x_sample[:, :, i] + self._sigma * dw_sample[:, :, i]
#            return dw_sample, x_sample
#
#        dw_sample, x_sample = sample(1)
#
#        if self.done:
#            y = y - self._bsde.delta_t * self._bsde.f_tf(
#                time_stamp[-1], x[:, :, -2], y, z
#            ) + torch.sum(z * dw[:, :, -1], 1, keepdim=True)
#            delta = y - self._bsde.g_tf(self._total_time, x[:, :, -1])
#            rwd = np.abs(delta)
#        else:
#            rwd = 0
#
#        return x_sample, rwd, self.done, 0

