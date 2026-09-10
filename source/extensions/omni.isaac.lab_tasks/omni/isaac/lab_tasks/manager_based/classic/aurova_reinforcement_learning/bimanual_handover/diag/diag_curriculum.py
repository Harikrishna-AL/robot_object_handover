"""
Verifies the reverse-curriculum reset placement BEFORE spending a training run
on it. For each curriculum level it forces that level, resets, and measures
where the receiver's hand actually ends up relative to the object.

Checks two distinct things:
  1. POSITION -- does level 0 really start the hand on the object, and does the
     distance grow monotonically with level? If not, the IK placement or the
     analytic object pose in _apply_curriculum_reset is wrong.
  2. ORIENTATION -- the known limitation. Only position is interpolated, so the
     wrist keeps its sampled home orientation. If the hand arrives at the right
     place but pointing the wrong way, the fingertip SPREAD (max - min distance
     to the object) stays large even as the mean shrinks: the hand is beside
     the object rather than around it. That distinguishes "placement is broken"
     from "placement is fine but orientation needs work".

Uses zero actions and needs no checkpoint, so it can run immediately after a
container rebuild.
"""
import argparse, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--settle_steps', type=int, default=3)
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
recv = u.scene.articulations[u.cfg.keys[R]]
n_levels = len(u.cfg.curriculum_start_frac)

print(f"curriculum_start_frac = {list(u.cfg.curriculum_start_frac)}", flush=True)
print(f"\n{'lvl':>3} {'frac':>6} | {'tips->obj mean':>14} {'min':>7} {'max':>7} {'spread':>7} | {'stage':>5}", flush=True)
print("-" * 72, flush=True)

zero = torch.zeros((args_cli.num_envs, env.action_space.shape[-1]))
for level in range(n_levels):
    u.curriculum_level = level
    env.reset()
    # let the written joint state propagate before measuring
    for _ in range(args_cli.settle_steps):
        env.step(zero)

    obj = u.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]           # (N,3)
    tips = recv.data.body_state_w[:, u.finger_tips[R], 0:3]                     # (N,T,3)
    d = torch.norm(tips - obj.unsqueeze(1), dim=-1)                             # (N,T)
    mean_d = d.mean().item(); min_d = d.min(dim=-1).values.mean().item()
    max_d = d.max(dim=-1).values.mean().item(); spread = (max_d - min_d)
    stage = u.stage.float().mean().item()
    frac = u.cfg.curriculum_start_frac[level]
    print(f"{level:>3} {frac:>6.2f} | {mean_d:>14.4f} {min_d:>7.4f} {max_d:>7.4f} {spread:>7.4f} | {stage:>5.2f}", flush=True)

print("\nEXPECTED: mean distance rises with level; level 0 should be small (<~0.06 m).", flush=True)
print("If mean is small but spread is large, position is right and ORIENTATION is the gap.", flush=True)
print("DONE", flush=True)
simulation_app.close()
