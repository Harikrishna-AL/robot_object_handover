# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import torch
import copy

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--checkpoint_freq", type=int, default=500, help="Checkpoint interval in environment steps.")
parser.add_argument("--resume_checkpoint", type=str, required=True, help="SB3 checkpoint to resume.")
parser.add_argument("--resume_learning_rate", type=float, default=1e-5)
parser.add_argument("--resume_curriculum_level", type=int, default=4)
parser.add_argument("--train", type=bool, default=True, help="Wether to train or perform normal loop")
parser.add_argument("--vel_rew", action="store_true", default=False, help="Add velocity penalty reward term")
parser.add_argument("--use_force_rewards", action="store_true", default=False)
parser.add_argument("--use_signal_1", action="store_true", default=False)
parser.add_argument("--use_signal_2", action="store_true", default=False)
parser.add_argument("--use_signal_instability", action="store_true", default=False)
parser.add_argument("--F_ref", type=float, default=0.15)
parser.add_argument("--F_safe", type=float, default=0.35)
parser.add_argument("--F_threshold", type=float, default=0.10)
parser.add_argument("--lambda_balance", type=float, default=0.1)
parser.add_argument("--lambda_instability", type=float, default=0.1)
parser.add_argument("--lambda_force_excess", type=float, default=0.1)
parser.add_argument("--palm_weight", type=float, default=0.3)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
from datetime import datetime
from torch import nn

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_schedule_fn

class Sb3CustomMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            custom_keys = [k for k in self.model.ep_info_buffer[-1].keys() if k not in ["r", "l", "t"]]
            for key in custom_keys:
                values = [ep_info[key] for ep_info in self.model.ep_info_buffer if key in ep_info]
                if len(values) > 0:
                    self.logger.record(key, np.mean(values))

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from train_utils import AddNoiseObservation

import wandb
from wandb.integration.sb3 import WandbCallback

from networks import *


# directory for logging into
path_to_train = os.environ.get(
    "BIMANUAL_LOG_ROOT",
    "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train",
)
log_dir = os.path.join(path_to_train, "logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
run = wandb.init(project="bim_hand_dani_julio", name=log_dir.split("/")[-1], sync_tensorboard=True)


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    if hasattr(args_cli, "vel_rew"):
        env_cfg.vel_rew = args_cli.vel_rew

    env_cfg.use_force_rewards = args_cli.use_force_rewards
    env_cfg.use_signal_1 = args_cli.use_signal_1
    env_cfg.use_signal_2 = args_cli.use_signal_2
    env_cfg.use_signal_instability = args_cli.use_signal_instability
    env_cfg.F_ref = args_cli.F_ref
    env_cfg.F_safe = args_cli.F_safe
    env_cfg.F_threshold = args_cli.F_threshold
    env_cfg.lambda_balance = args_cli.lambda_balance
    env_cfg.lambda_instability = args_cli.lambda_instability
    env_cfg.lambda_force_excess = args_cli.lambda_force_excess
    env_cfg.palm_weight = args_cli.palm_weight


    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)


    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # The checkpoint was trained on the unmodified, hardest reset. Preserve
    # that task on resume: SB3 checkpoints do not contain IsaacLab env state.
    unwrapped = env.unwrapped
    max_level = len(unwrapped.cfg.curriculum_start_frac) - 1
    if not 0 <= args_cli.resume_curriculum_level <= max_level:
        raise ValueError(f"resume curriculum level must be in [0, {max_level}]")
    unwrapped._set_curriculum_level(args_cli.resume_curriculum_level)
    # Level 4 is the real midpoint task. Disable later demotion so a brief
    # post-resume dip cannot silently change the evaluation distribution.
    if args_cli.resume_curriculum_level == max_level:
        unwrapped.cfg.use_reverse_curriculum = False
    print(f"[RESUME] curriculum_level={unwrapped.curriculum_level}, "
          f"start_frac={unwrapped.cfg.curriculum_start_frac[unwrapped.curriculum_level]}", flush=True)

    # Add noise wrappers
    # env = gym.wrappers.TransformObservation(env, lambda obs: obs["policy"] + 1000 * torch.rand(obs["policy"].shape).to(obs["policy"].device), env.observation_space)
    #env = AddNoiseObservation(env, noise_std=0.1)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # Resolve the policy class from the yaml's `policy:` string (e.g.
    # 'CustomActorCriticPolicy' or 'SharedArmActorCriticPolicy', both defined
    # in networks.py and already in scope via `from networks import *`)
    # instead of hardcoding a single class -- lets sb3_ppo_cfg_bidirectional.yaml
    # select SharedArmActorCriticPolicy without a second copy of this script.
    policy_cls = globals().get(policy_arch, CustomActorCriticPolicy)

    # create agent from stable baselines
    agent = PPO.load(args_cli.resume_checkpoint, env=env, device=agent_cfg.get("device", "cpu"))
    if not isinstance(agent.policy, policy_cls):
        raise TypeError(f"checkpoint policy {type(agent.policy)} != expected {policy_cls}")
    agent.learning_rate = args_cli.resume_learning_rate
    agent.lr_schedule = get_schedule_fn(args_cli.resume_learning_rate)
    agent.target_kl = 0.02
    print(f"[RESUME] checkpoint={args_cli.resume_checkpoint}, "
          f"loaded_timesteps={agent.num_timesteps}, "
          f"learning_rate={agent.learning_rate}, target_kl={agent.target_kl}", flush=True)

    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=args_cli.checkpoint_freq, save_path=log_dir, name_prefix="model", verbose=2)
    
    if args_cli.train:
        # train the agent
        agent.learn(total_timesteps=max(1, int(n_timesteps) - agent.num_timesteps),
                    reset_num_timesteps=False,
                    callback=[checkpoint_callback, Sb3CustomMetricsCallback(), WandbCallback()])
        # save the final model
        agent.save(os.path.join(log_dir, "model"))

    else:
        obs = env.reset()
                             
        action = torch.rand((env_cfg.scene.num_envs, 6+4))
        action = torch.tensor([[0, 0, 0, 0, 0, 0, 
                                 0, 0, 0]])

        # Simulate physics
        while simulation_app.is_running():
            with torch.inference_mode():

                # Step the environment
                obs, reward, done, info = env.step(action)


    run.finish()  # stop uploading data to wandb

    env.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
