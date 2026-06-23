from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

from .utils import TensorQueue


def obs_test(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(obs_test, "sequence"):
        obs_test.sequence = TensorQueue(max_size=5, element_shape=(asset.num_instances, 13))
    
    curr_pos_w = asset.data.root_state_w
    obs_test.sequence.enqueue(curr_pos_w)

    return curr_pos_w