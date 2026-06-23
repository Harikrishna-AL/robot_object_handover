import argparse
# from pynput import keyboard

from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="UR5e RL environment")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="Isaac-UR5e-joint-reach-v0", help="Name of the task.")  # Isaac-UR5e-reach-v0

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows"""

import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils import parse_env_cfg
import gymnasium as gym
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry
import math
from scipy.spatial.transform import Rotation
# from pynput import keyboard

key_cmd = '\n'
pos_cmd = "sdp869"
neg_cmd = "wal241"
inc = 0.01
action = torch.tensor([[-0.6880,  0.1639,  0.6471, -0.2706,  0.6533, -0.2706, -0.6533]])
action = torch.tensor([-0.4925,  0.1338,  0.4810, 1, -0.3832, -0.9236, -0.0028] + [0.0]*4 + [1]*4 + [2]*4 + [3]*4) # -0.0068, 1 en w por el angle scale
action = torch.zeros(6+16)
action[0] = 0.0
action[7] = 0.263
# action = torch.tensor([[-0.4925,  0.1338,  0.4810, 1,1,1,1]])
# action = torch.zeros((1,6))

def on_press(key):
    global key_cmd, action

    try:
        key_cmd = key.char
        action = update_cmd(action)
        
    except AttributeError:
        pass
        
def update_cmd(cmd):
    global key_cmd, pos_cmd, neg_cmd, inc

    if key_cmd in pos_cmd:
        list_cmds, mult = pos_cmd, inc

    elif key_cmd in neg_cmd:
        list_cmds, mult = neg_cmd, -inc

    else: 
        return cmd

    idx = list_cmds.index(key_cmd)
    key_cmd = '\n'

    if idx < 3:
        cmd[:, idx] += mult
    else:
        quat_df = cmd[:, 3:]
            
        rot = Rotation.from_quat(quat_df)
        rot_euler = rot.as_euler('xyz', degrees=True)

        print(rot_euler)
        
        rot_euler[:, idx - 3] += mult/inc

        rot = Rotation.from_euler(angles = rot_euler, seq='xyz', degrees=True)
        rot_quat = rot.as_quat()

        cmd[:, 3:] = torch.tensor(rot_quat)


    return cmd


def main():
    """Main function"""
    
    # listener = keyboard.Listener(
    #     on_press=on_press)
    # listener.start()


    # Create environment configuration
    # env_cfg = UR5eRLReachCfg()
    # env_cfg.scene.num_envs = args_cli.num_envs

    # # Setup RL environment
    # env = ManagerBasedRLEnv(cfg=env_cfg)

    env_cfg = parse_env_cfg(
        task_name = args_cli.task, device = args_cli.device, num_envs = args_cli.num_envs, use_fabric = not args_cli.disable_fabric
    )

    env = gym.make(args_cli.task, cfg = env_cfg)
    
    # Reset
    env.reset()

    # Simulate physics
    while simulation_app.is_running():
        with torch.inference_mode():
            # Step the environment
            # command = env.command_manager.get_command("ee_pose")[0].cpu()
            
            
            # w = command[3]
            # command[3], command[4], command[5] = command[4], command[5], command[6]
            # command[6] = w
            
            
            # rot = Rotation.from_quat(command[3:])
            # rot_euler = torch.tensor(rot.as_euler(seq = "xyz" ,degrees = False))

            # new_cmd = torch.cat((command[0:3], rot_euler)).unsqueeze(0)


            # obs, rew, terminated, truncated, info = env.step(env.command_manager.get_command("ee_pose"))
            # update_cmd(env.command_manager.get_command("ee_pose"))
            # print()
            # print("AAAAAA") torch.zeros(env.action_space.sample().shape)

            obs, rew, terminated, truncated, info = env.step(action.repeat(args_cli.num_envs,1))
            # print()


    env.close()


if __name__ == "__main__":
    # Run the main the function
    main()

    # Close sim app
    simulation_app.close()