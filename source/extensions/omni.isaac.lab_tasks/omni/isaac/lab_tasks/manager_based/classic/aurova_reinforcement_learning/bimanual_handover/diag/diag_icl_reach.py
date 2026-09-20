"""
Verify the reach curriculum before committing a 48h run to it.

For each level: force it, reset, and report the receiver's fingertip distance to
the object. Must be small at level 0 and rise monotonically to the home-pose
distance (~0.42m) at the final level.

Also reports the object's ENV-LOCAL positional spread, which must stay roughly
CONSTANT across levels -- that is the property the previous curriculum broke by
annealing the randomisation itself, collapsing std 1.00 -> 0.12 in 25M steps.
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
R = u.cfg.GEN3
art = u.scene.articulations[u.cfg.keys[R]]
n_levels = len(u.cfg.icl_reach_frac)
zero = torch.zeros((args_cli.num_envs, env.action_space.shape[-1]))

print(f"icl_reach_frac = {list(u.cfg.icl_reach_frac)}", flush=True)
print(f"\n{'lvl':>3} {'frac':>6} | {'tips->obj (m)':>14} {'min':>7} {'max':>7} | {'obj spread (m)':>14}", flush=True)
print("-" * 66, flush=True)

for level in range(n_levels):
    u.icl_level = level
    env.reset()
    env.step(zero)
    tips = art.data.body_state_w[:, u.finger_tips[R], 0:3].mean(1)
    obj = u.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]
    d = torch.norm(tips - obj, dim=-1)
    spread = (obj - u.scene.env_origins).std(dim=0).mean().item()
    print(f"{level:>3} {u.cfg.icl_reach_frac[level]:>6.2f} | {d.mean().item():>14.4f} "
          f"{d.min().item():>7.4f} {d.max().item():>7.4f} | {spread:>14.4f}", flush=True)

print("\nEXPECTED: distance RISES with level (small at 0, ~0.42m at the last).", flush=True)
print("EXPECTED: object spread stays ROUGHLY CONSTANT -- randomisation is not annealed.", flush=True)
print("DIAG_ICL_REACH_DONE", flush=True)
simulation_app.close()
