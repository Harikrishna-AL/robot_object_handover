import gymnasium as gym

from .bimanual_direct_env_cfg import BimanualDirectCfg
from .bimanual_bidirectional_env_cfg import BimanualBidirectionalCfg
from .bimanual_direct_icl_env_cfg import BimanualDirectICLCfg
from . import agents


gym.register(
    id="Isaac-Bimanual-Direct-reach-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.bimanual_handover.bimanual_direct_env:BimanualDirect",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BimanualDirectCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml"
    }
)

# Baseline task + inverse curriculum over the holder's pose-randomisation
# range. Registered separately so BimanualDirect stays untouched and a
# with/without-ICL comparison is apples-to-apples. See
# bimanual_direct_icl_env_cfg.py for what is annealed and why.
gym.register(
    id="Isaac-Bimanual-Direct-reach-ICL-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.bimanual_handover.bimanual_direct_icl_env:BimanualDirectICL",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BimanualDirectICLCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml"
    }
)

# Round-trip handover: GEN3 grasps the object from UR5e, then UR5e grasps it
# back from GEN3, both driven by one weight-shared policy. See
# bimanual_bidirectional_env.py's module docstring for the design.
gym.register(
    id="Isaac-Bimanual-Bidirectional-v0",
    entry_point="omni.isaac.lab_tasks.manager_based.classic.aurova_reinforcement_learning.bimanual_handover.bimanual_bidirectional_env:BimanualBidirectional",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BimanualBidirectionalCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg_bidirectional.yaml",
    }
)
