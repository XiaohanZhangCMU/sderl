# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms
from sderl.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
from sderl.algos.tf1.ppo.ppo import ppo as ppo_tf1
from sderl.algos.tf1.sac.sac import sac as sac_tf1
from sderl.algos.tf1.td3.td3 import td3 as td3_tf1
from sderl.algos.tf1.trpo.trpo import trpo as trpo_tf1
from sderl.algos.tf1.vpg.vpg import vpg as vpg_tf1

from sderl.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from sderl.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from sderl.algos.pytorch.dppo.dppo import dppo as dppo_pytorch
from sderl.algos.pytorch.sac.sac import sac as sac_pytorch
from sderl.algos.pytorch.td3.td3 import td3 as td3_pytorch
from sderl.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from sderl.algos.pytorch.vpg.vpg import vpg as vpg_pytorch
from sderl.algos.pytorch.dvpg.dvpg import dvpg as dvpg_pytorch

# Loggers
from sderl.utils.logx import Logger, EpochLogger

# Version
from sderl.version import __version__
