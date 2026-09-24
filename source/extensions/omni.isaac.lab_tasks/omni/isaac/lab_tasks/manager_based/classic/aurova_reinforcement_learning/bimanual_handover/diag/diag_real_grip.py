"""
Does the giving hand actually HOLD the object once its fingers are closed?

Runs with the kinematic weld OFF and both arms commanded still, then watches the
object for 300 steps. This is the same zero-action control that caught the
sinking-object bug: with the arms motionless, anything that happens to the
object is physics, not policy.

PASS  obj_z holds ~constant, giver fingertips stay close, contact force > 0
FAIL  obj_z sinks (the grip is not holding) -- most likely the object is not
      positioned inside the closed fingers, since obj_pos_trans was tuned for
      the UR5e's ee_link, not the GEN3's end_effector_link.
"""
import argparse, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--weld', type=str, default='off', choices=['on', 'off'])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa

env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.use_object_weld = (args_cli.weld == 'on')
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped
G = u.cfg.UR5e                       # giver
giver = u.scene.articulations[u.cfg.keys[G]]
n_arm = len(u.cfg.joints[G])

print(f"weld = {args_cli.weld}   object_height_limit = {u.cfg.object_height_limit:.4f}", flush=True)
print(f"giver closed-hand joints (first 8): "
      f"{[round(float(x),3) for x in u.closed_hand_joints[0, :8].tolist()]}", flush=True)

obs, _ = env.reset()
zero = torch.zeros((args_cli.num_envs, env.action_space.shape[-1]))
print(f"\n{'step':>5} {'obj_z':>7} {'tips->obj':>10} {'giver_contact':>14} {'hand_j1':>8}", flush=True)
z0 = None
for step in range(args_cli.steps):
    obs, r, te, tr, info = env.step(zero)
    obj = u.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]
    tips = giver.data.body_state_w[:, u.finger_tips[G], 0:3].mean(1)
    d = torch.norm(tips - obj, dim=-1).mean().item()
    contact = u.contacts[:, u._ur5e_contact_slice].sum(-1).mean().item()
    hj = giver.data.joint_pos[0, n_arm + 1].item()     # a curled finger joint
    if z0 is None: z0 = obj[:, 2].mean().item()
    if step < 6 or step % 30 == 0 or step == args_cli.steps - 1:
        print(f"{step:>5} {obj[:,2].mean().item():>7.3f} {d:>10.4f} {contact:>14.3f} {hj:>8.3f}", flush=True)

zf = obj[:, 2].mean().item()
print(f"\nobj_z: {z0:.4f} -> {zf:.4f}   drop = {z0-zf:.4f} m over {args_cli.steps} steps", flush=True)
print(f"VERDICT: {'HELD' if (z0 - zf) < 0.02 else 'SLIPPED'}", flush=True)
print("DIAG_REAL_GRIP_DONE", flush=True)
simulation_app.close()
