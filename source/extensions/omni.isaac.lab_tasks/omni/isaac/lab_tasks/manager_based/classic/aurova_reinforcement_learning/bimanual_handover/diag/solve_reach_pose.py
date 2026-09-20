"""
Solve the receiver's joint configuration that puts its hand ON the object, once,
offline -- so the curriculum can interpolate toward it at reset without running
IK during training.

Why offline: _reset_idx runs outside the physics loop, so stepping the sim there
would advance every environment including the ones not resetting. A standalone
script has no episodes in flight and can step freely.

Why per-iteration printing: three previous attempts at this failed silently or
diverged (single differential step moved the arm ~5cm toward a 0.3m target;
iterating it in __init__ walked the arm 0.64m AWAY because body_state_w never
refreshed between iterations). This prints the fingertip->object distance at
every iteration under three different refresh strategies, so a non-converging
solve is obvious rather than something discovered 48 hours later.
"""
import argparse, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Direct-reach-v0')
parser.add_argument('--iters', type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils.math import combine_frame_transforms, subtract_frame_transforms
import omni.isaac.lab_tasks  # noqa

env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=1)
# Pin the object to its NOMINAL pose so the solved configuration is a single
# well-defined target. Training keeps the full +/-0.15m randomisation.
env_cfg.apply_range = [False, False]
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped
R = u.cfg.GEN3
art = u.scene.articulations[u.cfg.keys[R]]
n_arm = len(u.cfg.joints[R])
dt = u.sim.get_physics_dt() if hasattr(u.sim, 'get_physics_dt') else 1.0 / 120.0

env.reset()

def grasp_point_world():
    obj = u.scene.rigid_objects["object"].data.body_state_w[:, 0, 0:7]
    return combine_frame_transforms(
        t01=obj[:, :3], q01=obj[:, 3:],
        t12=u.cfg.grasp_obs_obj_pos_trans, q12=u.cfg.grasp_obs_obj_quat_trans)[0]

def tips_world():
    return art.data.body_state_w[:, u.finger_tips[R], 0:3].mean(1)

def dist_to_object():
    return torch.norm(tips_world() - grasp_point_world(), dim=-1).mean().item()

home_q = art.data.joint_pos.clone()
print(f"START fingertips->object = {dist_to_object():.4f} m", flush=True)

def solve(refresh, iters):
    """refresh: how to make body_state_w reflect a freshly written joint state."""
    art.write_joint_state_to_sim(home_q, torch.zeros_like(home_q), None, None)
    u.sim.step(render=False); u.scene.update(dt)
    traj = []
    q = home_q.clone()
    best_d, best_q = float('inf'), home_q.clone()
    for it in range(iters):
        gp_w = grasp_point_world()
        root = art.data.root_state_w[:, 0:7]
        gp_r, _ = subtract_frame_transforms(t01=root[:, :3], q01=root[:, 3:],
                                            t02=gp_w, q02=art.data.body_state_w[:, u.ee_jacobi_idx[R] + 1, 3:7])
        ee_pos_r, ee_quat_r, jac, jp = u._get_ee_pose(R)
        target = torch.cat((gp_r, ee_quat_r), dim=-1)   # position target, hold current wrist orientation
        u.controller.set_command(target)
        arm_q = u.controller.compute(ee_pos_r, ee_quat_r, jac, jp)
        q = torch.cat((arm_q, home_q[:, n_arm:]), dim=-1)
        art.write_joint_state_to_sim(q, torch.zeros_like(q), None, None)
        if refresh == 'sim_step':
            u.sim.step(render=False); u.scene.update(dt)
        elif refresh == 'update_only':
            art.update(dt)
        elif refresh == 'env_step':
            env.step(torch.zeros((1, env.action_space.shape[-1])))
        d = dist_to_object()
        traj.append(d)
        if d < best_d:
            best_d, best_q = d, q.clone()
    return best_q, traj, best_d

_candidates = []
for refresh in ('sim_step', 'update_only', 'env_step'):
    try:
        q, traj, best_d = solve(refresh, args_cli.iters)
        nums = ' '.join(f"{d:.4f}" for d in traj)
        print(f"\n[{refresh}] dist per iter: {nums}", flush=True)
        print(f"[{refresh}] BEST dist = {best_d:.4f} m at iter {traj.index(best_d)+1}   (last={traj[-1]:.4f})   converged={best_d < 0.08}", flush=True)
        if best_d < 0.08:
            arm = [round(float(x), 6) for x in q[0, :n_arm].tolist()]
            print(f"[{refresh}] SOLVED_ARM_Q = {arm}", flush=True)
            _candidates.append((refresh, q.clone()))
    except Exception as e:
        import traceback
        print(f"[{refresh}] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

# ---- verify the WRAPPED configuration ------------------------------------
# The raw solutions wind joints through multiple revolutions (job 4267693 gave
# 7.5 and -13.0 rad). q and q+2*pi*k are kinematically identical for a revolute
# joint, so wrapping into [-pi, pi] should preserve the fingertip position --
# but interpolating toward an unwrapped value would sweep the arm through whole
# revolutions at intermediate curriculum levels. Verify empirically rather than
# assume, and check the result lies inside the joint limits.
import math
print("\n==== WRAPPED VERIFICATION ====", flush=True)
lims = None
for attr in ('soft_joint_pos_limits', 'joint_pos_limits', 'joint_limits'):
    lims = getattr(art.data, attr, None)
    if lims is not None:
        print(f"joint limits from art.data.{attr}", flush=True); break

for name, q in _candidates:
    qw = q.clone()
    qw[:, :n_arm] = torch.remainder(q[:, :n_arm] + math.pi, 2 * math.pi) - math.pi
    art.write_joint_state_to_sim(qw, torch.zeros_like(qw), None, None)
    u.sim.step(render=False); u.scene.update(dt)
    d_wrapped = dist_to_object()
    arm = [round(float(x), 6) for x in qw[0, :n_arm].tolist()]
    inside = "unknown"
    if lims is not None:
        lo = lims[0, :n_arm, 0]; hi = lims[0, :n_arm, 1]
        ok = bool(((qw[0, :n_arm] >= lo) & (qw[0, :n_arm] <= hi)).all())
        inside = "YES" if ok else "NO"
    print(f"[{name}] wrapped dist = {d_wrapped:.4f} m   within_limits={inside}", flush=True)
    print(f"[{name}] WRAPPED_ARM_Q = {arm}", flush=True)

print("\nSOLVE_DONE", flush=True)
simulation_app.close()
