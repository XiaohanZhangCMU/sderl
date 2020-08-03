# Credits for:
#     TF-DQN: github.com/axitkhurana/spinningup/spinup/algos/dqn
#     Pytorch-DQN: github.com/kashif/firedup/firedup/algos/dqn

from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import gym
import time
import sderl.algos.pytorch.dqn.core as core
from sderl.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch= dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

"""

Deep Q Network (DQN)

"""
def dqn(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=500, epochs=100, replay_size=int(1e6), gamma=0.99,
        epsilon_start=1, epsilon_step=1e-4, epsilon_end=0.1,
        lr=1e-3, batch_size=100, start_steps=5000,
        max_ep_len=1000, logger_kwargs=dict(), update_freq=100, save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``q``        (batch,)          | Gives the current estimate
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        epsilon_start (float): Starting value of epsilon (probability with which
            we take random action, always between 0 and 1)

        epsilon_step (float): Reduce epsilon by this amount every step.

        epsilon_end (float): Stop at this value of epsilon.

        lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        update_freq (int): How often (in terms of gap between epochs) to update
            the parameters of target network

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_target = deepcopy(ac)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.q])
    logger.log('\nNumber of parameters: \t q: %d\n'%var_counts)

    q_params = itertools.chain(ac.q.parameters(), ac_target.q.parameters())
    q_optimizer = Adam(q_params, lr=lr)

    def update(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o)

        one_hot_act = torch.nn.functional.one_hot(a.long(), env.action_space.n)
        qa_theta = torch.sum(one_hot_act*q, axis=1)
        qa_theta = qa_theta.float()
        with torch.no_grad():
            q_target = ac_target.q(o2)
            tmp,_ = torch.max(q_target, 1)
            qa_target = r + gamma*(1-d)*tmp

        # DQN losses
        #q_loss = torch.mean((qa_theta-qa_target)**2)
        q_loss = F.smooth_l1_loss(qa_theta, qa_target)

        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        q_info = dict(QVals=q.cpu().detach().numpy())
        logger.store(LossQ=q_loss.item(), **q_info)


    # Setup model saving
    logger.setup_pytorch_saver(ac)

    def get_action(o, epsilon):
        """Select an action from the set of available actions.
        Chooses an action randomly with probability epsilon otherwise
        act greedily according to the current Q-value estimates.
        """
        if np.random.random() <= epsilon:
            return env.action_space.sample()
        else:
            q_values = ac.q(torch.Tensor(o.reshape(1, -1)))
            return torch.argmax(q_values, dim=1).item()

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Anneal epsilon linearly from epsilon_start with epsilon_step
        With epsilon probabilty we choose a random action for
        better exploration.
        """

        ac.q.eval()

        epsilon = epsilon_start - (t * epsilon_step)
        if epsilon < epsilon_end:
            epsilon = epsilon_end
        a = get_action(o, epsilon)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Store episode return at the end of trajectory
            """
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t % update_freq == 0:
            # Copy parameters operation between theta & target
            ac_target.q.load_state_dict(ac.q.state_dict())

        if t > start_steps:
            ac.q.train()
            data = replay_buffer.sample_batch(batch_size)

            update(data)

        # End of epoch wrap-up
        if t > start_steps and (t - start_steps) % steps_per_epoch == 0:
            epoch = (t - start_steps) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Epsilon', epsilon)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='dqn')
    args = parser.parse_args()

    from sderl.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
