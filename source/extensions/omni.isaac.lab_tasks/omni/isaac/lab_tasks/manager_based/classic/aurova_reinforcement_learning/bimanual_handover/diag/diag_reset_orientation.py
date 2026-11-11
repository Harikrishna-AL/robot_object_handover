"""Generate Cartesian reset candidates using actual palm surface normals.

The outward normal is hand_palm_link +X, established by the positive-X palm
contact plates in the asset. Earlier versions used mean fingertip minus palm;
that is a finger-extension proxy and their reported angles were not palm angles.
This static search does not prove grip stability. Validate candidates with
weld-free physical holding in both roles and joint/collision checks.
"""
import argparse, math, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--coarse', type=int, default=6, help='grid points per axis, coarse pass')
parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'search6'],
                    help="'grid' sweeps orientation only; 'search6' does a CONSTRAINED random "
                         "search over position AND orientation")
parser.add_argument('--samples', type=int, default=600, help='search6: random samples')
parser.add_argument('--pos_delta', type=float, default=0.18, help='search6: +/- metres on x,y,z')
parser.add_argument('--z_margin', type=float, default=0.15,
                    help='object must spawn at least this far above object_height_limit')
parser.add_argument('--grip_tol', type=float, default=0.03,
                    help='giver fingertips->object may not grow more than this vs baseline')
parser.add_argument('--settle', type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, torch
from omni.isaac.lab.utils.math import quat_from_euler_xyz, quat_apply
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa

env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.randomize_giver = False
env_cfg.reset_arm_joint_pos = None
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped
U, G3 = u.cfg.UR5e, u.cfg.GEN3

PALM = 'hand_palm_link'
palm_idx = {}
for idx in (U, G3):
    found = u.scene.articulations[u.cfg.keys[idx]].find_bodies(PALM)[0]
    assert len(found) == 1
    palm_idx[idx] = found[0]

base_rpy = u.ee_pose_ranges[:, 3:6, 0].clone()   # [2, 3]; min==max when apply_range is False
print(f"baseline reset rpy  UR5e={[round(float(x),4) for x in base_rpy[U]]}  "
      f"GEN3={[round(float(x),4) for x in base_rpy[G3]]}", flush=True)

def measure():
    """(worse_angle_deg, ur5e_deg, gen3_deg, obj_z, giver_tips_to_obj)."""
    env.reset()
    for _ in range(args_cli.settle):
        u.sim.step(render=False)
        u.scene.update(u.sim.get_physics_dt())
    pos, gdir = {}, {}
    for idx in (U, G3):
        art = u.scene.articulations[u.cfg.keys[idx]]
        p = art.data.body_state_w[:, palm_idx[idx], 0:3]
        normal = torch.tensor([1.0, 0.0, 0.0], device=u.device).expand(u.num_envs, -1)
        pos[idx] = p
        gdir[idx] = quat_apply(art.data.body_state_w[:, palm_idx[idx], 3:7], normal)
    ang = {}
    for idx, other in ((U, G3), (G3, U)):
        to_other = torch.nn.functional.normalize(pos[other] - pos[idx], dim=-1)
        cos = (gdir[idx] * to_other).sum(-1).clamp(-1.0, 1.0)
        ang[idx] = torch.rad2deg(torch.acos(cos)).mean().item()
    obj = u.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]
    tips_giver = u.scene.articulations[u.cfg.keys[U]].data.body_state_w[
        :, u.finger_tips[U], 0:3].mean(1)
    d_give = torch.norm(tips_giver - obj, dim=-1).mean().item()
    return max(ang[U], ang[G3]), ang[U], ang[G3], obj[:, 2].mean().item(), d_give

base_pos = u.ee_pose_ranges[:, 0:3, 0].clone()

def set_rpy(delta):
    '''Apply the same (dr, dp, dy) delta to BOTH arms' root-relative target.'''
    for idx in (U, G3):
        for k in range(3):
            u.ee_pose_ranges[idx, 3 + k, 0] = base_rpy[idx, k] + delta[k]
            u.ee_pose_ranges[idx, 3 + k, 1] = base_rpy[idx, k] + delta[k]

def set_pose(dpos, drpy):
    """Apply root-relative position and orientation deltas to both arms.

    The parent's single differential-IK reset step couples achieved position
    and orientation. Geometry candidates still need dynamic grip validation.
    """
    for idx in (U, G3):
        for k in range(3):
            u.ee_pose_ranges[idx, k, 0] = base_pos[idx, k] + dpos[k]
            u.ee_pose_ranges[idx, k, 1] = base_pos[idx, k] + dpos[k]
            u.ee_pose_ranges[idx, 3 + k, 0] = base_rpy[idx, k] + drpy[k]
            u.ee_pose_ranges[idx, 3 + k, 1] = base_rpy[idx, k] + drpy[k]

w0, a0u, a0g, z0, d0 = measure()
print(f"\n=== BASELINE ===", flush=True)
print(f"  worse={w0:6.1f} deg  UR5e={a0u:6.1f}  GEN3={a0g:6.1f}  obj_z={z0:.4f}  giver_tips_to_obj={d0:.4f}", flush=True)

if args_cli.mode == 'search6':
    import random
    random.seed(0)
    z_min = u.cfg.object_height_limit + args_cli.z_margin
    d_max = d0 + args_cli.grip_tol
    print(f"\n=== CONSTRAINED SEARCH (6-DOF, {args_cli.samples} samples) ===", flush=True)
    print(f"  constraints: obj_z >= {z_min:.4f}   giver_tips_to_obj <= {d_max:.4f}", flush=True)
    bestS, feasible = None, 0
    for t in range(args_cli.samples):
        dpos = [random.uniform(-args_cli.pos_delta, args_cli.pos_delta) for _ in range(3)]
        drpy = [random.uniform(-math.pi, math.pi) for _ in range(3)]
        set_pose(dpos, drpy)
        w, au, ag, z, d = measure()
        if z < z_min or d > d_max:
            continue
        feasible += 1
        if bestS is None or w < bestS[0]:
            bestS = (w, au, ag, z, d, tuple(dpos), tuple(drpy))
            print(f"  [{t:4d}] worse={w:6.1f}  UR5e={au:5.1f} GEN3={ag:5.1f}  "
                  f"obj_z={z:.3f} grip_d={d:.3f}  dpos=({dpos[0]:+.3f},{dpos[1]:+.3f},{dpos[2]:+.3f}) "
                  f"drpy=({drpy[0]:+.3f},{drpy[1]:+.3f},{drpy[2]:+.3f})", flush=True)
    print(f"  feasible samples: {feasible}/{args_cli.samples}", flush=True)
    if bestS is None:
        print("  NO FEASIBLE CANDIDATE -- palms-facing is incompatible with the object "
              "height and grip constraints under this reset scheme.", flush=True)
    else:
        # local refinement around the best feasible point
        w, au, ag, z, d, dpos, drpy = bestS
        for _ in range(args_cli.samples // 2):
            p2 = [dpos[k] + random.uniform(-0.03, 0.03) for k in range(3)]
            r2 = [drpy[k] + random.uniform(-0.15, 0.15) for k in range(3)]
            set_pose(p2, r2)
            w2, au2, ag2, z2, d2 = measure()
            if z2 >= z_min and d2 <= d_max and w2 < w:
                w, au, ag, z, d, dpos, drpy = w2, au2, ag2, z2, d2, tuple(p2), tuple(r2)
                print(f"  refine worse={w:6.1f} obj_z={z:.3f} grip_d={d:.3f}", flush=True)
        print(f"\n=== RESULT (constrained) ===", flush=True)
        print(f"  baseline worse-of-two = {w0:.1f} deg  ->  best feasible = {w:.1f} deg", flush=True)
        print(f"  UR5e {a0u:.1f} -> {au:.1f},  GEN3 {a0g:.1f} -> {ag:.1f}", flush=True)
        print(f"  object height {z0:.4f} -> {z:.4f}  (limit {u.cfg.object_height_limit:.4f})", flush=True)
        print(f"  giver fingertips->object {d0:.4f} -> {d:.4f}", flush=True)
        print(f"\n  PASTE INTO cfg.ee_init_pose_quat:", flush=True)
        for idx, label in ((U, 'UR5e slot'), (G3, 'GEN3 slot')):
            rpy = base_rpy[idx] + torch.tensor(list(drpy), device=base_rpy.device)
            q = quat_from_euler_xyz(rpy[0:1], rpy[1:2], rpy[2:3])[0]
            pos = base_pos[idx] + torch.tensor(list(dpos), device=base_pos.device)
            print(f"    # {label}", flush=True)
            print(f"    [{float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f}, "
                  f"{float(q[0]):.4f}, {float(q[1]):.4f}, {float(q[2]):.4f}, {float(q[3]):.4f}],", flush=True)
    print("DIAG_RESET_ORIENTATION_DONE", flush=True)
    simulation_app.close()
    sys.exit(0)

best = (0.0, 0.0, 0.0, w0, a0u, a0g, z0, d0)
n = args_cli.coarse
print(f"\n=== COARSE PASS ({n}^3 = {n**3} resets over full rpy deltas) ===", flush=True)
for i in range(n):
    dr = -math.pi + 2 * math.pi * i / n
    for j in range(n):
        dp = -math.pi + 2 * math.pi * j / n
        for k in range(n):
            dy = -math.pi + 2 * math.pi * k / n
            set_rpy((dr, dp, dy))
            w, au, ag, z, d = measure()
            if w < best[3]:
                best = (dr, dp, dy, w, au, ag, z, d)
                print(f"  new best worse={w:6.1f}  d=({dr:+.3f},{dp:+.3f},{dy:+.3f})  "
                      f"UR5e={au:5.1f} GEN3={ag:5.1f} obj_z={z:.3f} grip_d={d:.3f}", flush=True)

print(f"\n=== REFINE around best (+/-0.25 rad, 5 per axis) ===", flush=True)
c = best[:3]
for i in range(5):
    dr = c[0] - 0.25 + 0.125 * i
    for j in range(5):
        dp = c[1] - 0.25 + 0.125 * j
        for k in range(5):
            dy = c[2] - 0.25 + 0.125 * k
            set_rpy((dr, dp, dy))
            w, au, ag, z, d = measure()
            if w < best[3]:
                best = (dr, dp, dy, w, au, ag, z, d)
                print(f"  new best worse={w:6.1f}  d=({dr:+.3f},{dp:+.3f},{dy:+.3f})  "
                      f"UR5e={au:5.1f} GEN3={ag:5.1f} obj_z={z:.3f} grip_d={d:.3f}", flush=True)

dr, dp, dy, w, au, ag, z, d = best
print(f"\n=== RESULT ===", flush=True)
print(f"  baseline worse-of-two = {w0:.1f} deg   ->   best = {w:.1f} deg", flush=True)
print(f"  UR5e {a0u:.1f} -> {au:.1f} deg,  GEN3 {a0g:.1f} -> {ag:.1f} deg", flush=True)
print(f"  rpy delta = ({dr:+.4f}, {dp:+.4f}, {dy:+.4f}) rad", flush=True)
print(f"  object height {z0:.4f} -> {z:.4f}  (object_height_limit = {u.cfg.object_height_limit:.4f})", flush=True)
print(f"  giver fingertips->object {d0:.4f} -> {d:.4f}  (grip must survive the rotation)", flush=True)
print(f"\n  PASTE INTO cfg.ee_init_pose_quat (position unchanged, orientation replaced):", flush=True)
for idx, label in ((U, 'UR5e slot'), (G3, 'GEN3 slot')):
    rpy = base_rpy[idx] + torch.tensor([dr, dp, dy], device=base_rpy.device)
    q = quat_from_euler_xyz(rpy[0:1], rpy[1:2], rpy[2:3])[0]
    pos = u.ee_pose_ranges[idx, 0:3, 0]
    print(f"    # {label}", flush=True)
    print(f"    [{float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f}, "
          f"{float(q[0]):.4f}, {float(q[1]):.4f}, {float(q[2]):.4f}, {float(q[3]):.4f}],", flush=True)
print("DIAG_RESET_ORIENTATION_DONE", flush=True)
simulation_app.close()
