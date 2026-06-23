import reach_rl
import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import VecNormalize, VecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import numpy as np
import os
import torch
from datetime import datetime


import wandb
from wandb.integration.sb3 import WandbCallback


env_id = "reach_rl/ReachRL-v0"
n_training_envs = 1
n_eval_envs = 1

# directory for logging into
# path_to_train = "/workspace/isaaclab/source/"
# log_dir = os.path.join(path_to_train, "logs", "reach_gym", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
# run = wandb.init(project="rl_manipulation_reach", name=log_dir.split("/")[-1], sync_tensorboard=True)


# --- Main ---
if __name__ == "__main__":
    print("|| Compiling ...")
    
    # Training environments
    vec_env  = make_vec_env(env_id, n_envs=n_training_envs, vec_env_cls = SubprocVecEnv, seed=0)

    n_steps = 16


    agent_cfg = {
        "seed": 14,

        # "n_timesteps": 3e8,      
        "policy": 'MlpPolicy',           
        "n_steps": n_steps,                   
        "batch_size": n_steps * n_training_envs,             
        "gae_lambda": 0.95,              
        "gamma": 0.99,                   
        "n_epochs": 20,                  
        "ent_coef": 0.01,                
        "learning_rate": 1e-3,
        "clip_range": 0.2,       
        "use_sde": False ,                
        "policy_kwargs": {
                        "activation_fn": torch.nn.Tanh,
                        "net_arch": [128, 64, 32],
                        "squash_output": False,
                        "share_features_extractor": True,},
        "vf_coef": 1.0,                  
        "max_grad_norm": 1.0,            
        "device": "cpu",
       
    }

    agent = PPO(env = vec_env, verbose=2, **agent_cfg)

    agent.learn(total_timesteps=1e6)
