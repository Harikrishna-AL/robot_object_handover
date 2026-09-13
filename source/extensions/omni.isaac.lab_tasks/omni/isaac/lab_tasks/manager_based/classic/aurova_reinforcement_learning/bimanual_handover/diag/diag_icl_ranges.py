"""
Confirms the inverse curriculum actually reaches the reset sampler.

For each level it forces that level, resets, and reports (a) the holder's
sampling interval width and (b) the ENV-LOCAL spread of the object's position
across environments. Both must be ~0 at level 0 and grow with the level.

Exists because the in-run self-check was initially written in world
coordinates, where the environment grid (~7.7m std) swamps the +/-0.15m the
curriculum varies, making it impossible to tell a working curriculum from a
no-op -- the same class of mistake that hid the self-play curriculum's
failure for three attempts.
"""
import argparse, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Direct-reach-ICL-v0')
parser.add_argument('--num_envs', type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa

env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped
H = u.cfg.UR5e
n_levels = len(u.cfg.icl_range_scale)

print(f"icl_range_scale = {list(u.cfg.icl_range_scale)}", flush=True)
print(f"\n{'lvl':>3} {'scale':>6} | {'xyz range width (m)':>20} | {'obj spread env-local (m)':>24}", flush=True)
print("-" * 62, flush=True)

zero = torch.zeros((args_cli.num_envs, env.action_space.shape[-1]))
for level in range(n_levels):
    u.icl_level = level
    u._apply_icl_level()
    env.reset()
    env.step(zero)
    width = (u.ee_pose_ranges[H, :3, 1] - u.ee_pose_ranges[H, :3, 0]).mean().item()
    obj_local = u.scene.rigid_objects["object"].data.body_state_w[:, 0, :3] - u.scene.env_origins
    spread = obj_local.std(dim=0).mean().item()
    print(f"{level:>3} {u.cfg.icl_range_scale[level]:>6.2f} | {width:>20.4f} | {spread:>24.4f}", flush=True)

print("\nEXPECTED: both columns ~0 at level 0 and increasing. Flat => the range", flush=True)
print("rescaling never reaches the sampler and any ICL run is meaningless.", flush=True)
print("DONE", flush=True)
simulation_app.close()
