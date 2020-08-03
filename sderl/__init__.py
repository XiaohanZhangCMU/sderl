# Algorithms

from sderl.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from sderl.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from sderl.algos.pytorch.dppo.dppo import dppo as dppo_pytorch
from sderl.algos.pytorch.sac.sac import sac as sac_pytorch
from sderl.algos.pytorch.sac_atari.sac import sac_atari as sac_atari_pytorch
from sderl.algos.pytorch.dqn.dqn import dqn as dqn_pytorch
from sderl.algos.pytorch.td3.td3 import td3 as td3_pytorch
from sderl.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from sderl.algos.pytorch.vpg.vpg import vpg as vpg_pytorch
from sderl.algos.pytorch.dvpg.dvpg import dvpg as dvpg_pytorch

# Loggers
from sderl.utils.logx import Logger, EpochLogger

# Version
from sderl.version import __version__
