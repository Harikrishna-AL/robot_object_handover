import argparse

from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="UR5e RL environment")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="Isaac-UR5e-joint-reach-v0", help="Name of the task.")
parser.add_argument("--model_dir", type=str, default="", help="Directory where the models are stored.")
parser.add_argument("--model_dir2", type=str, default="", help="Directory where the models are stored.")
parser.add_argument("--num_episodes", type=int, default=30, help="Number of steps per trial.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows"""

import torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
import gymnasium as gym
import os
from stable_baselines3 import PPO
import json


"""Main function"""
def main():

    # Parse environment configuration according to the task
    env_cfg = parse_env_cfg(
        task_name = args_cli.task, device = args_cli.device, num_envs = args_cli.num_envs, use_fabric = not args_cli.disable_fabric, 
    )
    
    # Environment creation
    env = gym.make(args_cli.task, cfg = env_cfg)
    
    # Filter models
    path_to_train = "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/logs/"
    
    dir = os.path.join(path_to_train, args_cli.model_dir)
    dir2 = os.path.join(path_to_train, args_cli.model_dir2)

    models = [file for file in os.listdir(dir) if file.endswith(".zip")]

    # Results of the evaluation
    results = {}
    traj = {}
    traj["distances"] = [[]]
    traj["phase"]     = [[]]

    # Reset
    obs, __ = env.reset()

    ep = 0

    # model_name_ =  "model_599552000_steps.zip"
    # model_name2_ = "model_599552000_steps.zip"

    model_name_ =  "model_27648000_steps.zip"
    model_name2_ = "model_27648000_steps.zip" # Update this too if evaluating 2 models!

    # Accumulated reward for all the episodes
    r = torch.zeros((args_cli.num_envs))

    # Loading model
    model = PPO.load(os.path.join(dir, model_name_))
    model.policy.eval()
    
    model2 = PPO.load(os.path.join(dir2, model_name2_))
    model2.policy.eval()


    # --- Loop through the models ---
    for idx, model_name in enumerate(models):
        if ep == args_cli.num_episodes: break

        print(f" -- Episode {ep+1}/{args_cli.num_episodes}")

        # Simulate physics
        while simulation_app.is_running():
            with torch.inference_mode():
                # Generate action
                # action, __ = model.predict(obs["policy"].cpu().numpy(), deterministic = True)
                action, __ = model2.predict(obs["policy"].cpu().numpy(), deterministic = True)
                
                # Step the environemnt
                obs, rew, terminated, truncated, info = env.step(torch.tensor((action)))
                
                traj["distances"][-1].append(obs["dist"])

                # Reset condition
                if terminated.item() or truncated.item():
                    
                    traj["distances"].append([])
                    print(f" -- Episode {ep+1}/{args_cli.num_episodes}")

                    ep+=1

                    # break
        
        # Compute mean reward
        mean_rew = torch.mean(r).item()

        print(f" ------ Reward per episode: {mean_rew}")

        # Create metric for one model
        results[model_name] = {
            "name": model_name,
            "mean_reward" : mean_rew,
        }


    # Serializing json
    json_object = json.dumps(traj, indent=4)
    
    saving_path = os.path.join(path_to_train, dir, "evaluation.json")

    # Writing to sample.json
    with open(saving_path, "w") as outfile:
        outfile.write(json_object)                    
                    


    env.close()


if __name__ == "__main__":
    # Run the main the function
    main()

    # Close sim app
    simulation_app.close()