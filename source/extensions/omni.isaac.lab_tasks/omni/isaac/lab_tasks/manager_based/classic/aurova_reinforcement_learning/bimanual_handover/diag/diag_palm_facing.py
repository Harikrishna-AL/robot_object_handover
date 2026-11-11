"""Measure actual palm-facing directions and sweep arm joints.

The front contact plates (palm_aux_link and palm_aux2_link) are thin in X and
located at positive X in hand_palm_link; its +X axis is the outward palm normal.
The old mean-fingertip-minus-palm vector measured finger extension instead.
With an open hand these axes can be nearly perpendicular, so previous proxy
angles and conclusions about wrist roll did not establish palm orientation.

Angle 0 means the palm surface faces the other hand; 180 means its back does.
A geometry sweep is only a candidate generator: verify real grip retention,
joint limits, and collisions before changing the environment.
"""
import argparse, math, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--settle', type=int, default=30, help='sim steps to settle after each write')
# Wrist roll can change the true palm normal even when the finger axis barely moves.
parser.add_argument('--joints', type=str, default='arm_joint_7,arm_joint_5',
                    help='comma-separated arm joint names to sweep individually')
parser.add_argument('--pairs', type=str, default='',
                    help="comma-separated 'jointA:jointB' to sweep jointly on a coarse grid")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils.math import quat_apply
import omni.isaac.lab_tasks  # noqa

env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.randomize_giver = False          # fixed layout; this is pure geometry
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped
U, G3 = u.cfg.UR5e, u.cfg.GEN3
dev = u.device
env.reset()

PALM = 'hand_palm_link'
palm_idx = {}
for idx in (U, G3):
    art = u.scene.articulations[u.cfg.keys[idx]]
    found = art.find_bodies(PALM)[0]
    assert len(found) == 1, f"{PALM} matched {len(found)} bodies on arm {idx}"
    palm_idx[idx] = found[0]
arm_names = u.cfg.joints[U]
j7 = arm_names.index('arm_joint_7')
j5 = arm_names.index('arm_joint_5')
print(f"palm body index: UR5e-slot={palm_idx[U]}  GEN3-slot={palm_idx[G3]}", flush=True)
print(f"arm_joint_5 col={j5}  arm_joint_7 col={j7}", flush=True)

def geom():
    """(angle_ur5e_deg, angle_gen3_deg, palm_separation) for the current state."""
    out = {}
    pos, gdir = {}, {}
    for idx in (U, G3):
        art = u.scene.articulations[u.cfg.keys[idx]]
        p = art.data.body_state_w[:, palm_idx[idx], 0:3]
        pos[idx] = p
        normal = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(u.num_envs, -1)
        gdir[idx] = quat_apply(art.data.body_state_w[:, palm_idx[idx], 3:7], normal)
    for idx, other in ((U, G3), (G3, U)):
        to_other = torch.nn.functional.normalize(pos[other] - pos[idx], dim=-1)
        cos = (gdir[idx] * to_other).sum(-1).clamp(-1.0, 1.0)
        out[idx] = torch.rad2deg(torch.acos(cos)).mean().item()
    sep = torch.norm(pos[U] - pos[G3], dim=-1).mean().item()
    return out[U], out[G3], sep

def hold_and_step(q):
    """Write both arms' full joint state, command it, and step physics so that
    body_state_w actually refreshes."""
    for idx in (U, G3):
        art = u.scene.articulations[u.cfg.keys[idx]]
        art.write_joint_state_to_sim(q[idx], torch.zeros_like(q[idx]))
        art.set_joint_position_target(q[idx])
    for _ in range(args_cli.settle):
        u.sim.step(render=False)
        u.scene.update(u.sim.get_physics_dt())

q0 = {idx: u.scene.articulations[u.cfg.keys[idx]].data.joint_pos.clone() for idx in (U, G3)}

hold_and_step(q0)
a_u, a_g, sep = geom()
print(f"\n=== BASELINE (as configured) ===", flush=True)
print(f"  palm separation            = {sep:.4f} m", flush=True)
print(f"  UR5e-slot (screen-LEFT)  palm normal vs toward-other = {a_u:6.1f} deg", flush=True)
print(f"  GEN3-slot (screen-RIGHT) palm normal vs toward-other = {a_g:6.1f} deg", flush=True)
print(f"  (0 = palm front faces the other arm, 180 = palm back faces it)", flush=True)

def sweep(col, label):
    print(f"\n=== SWEEP {label} (both arms set to the same value) ===", flush=True)
    print(f"  {'value(rad)':>11} {'deg':>7} {'UR5e_ang':>9} {'GEN3_ang':>9} {'worse_of_two':>13}", flush=True)
    best = None
    for k in range(16):
        v = -math.pi + 2 * math.pi * k / 16.0
        q = {idx: q0[idx].clone() for idx in (U, G3)}
        for idx in (U, G3):
            q[idx][:, col] = v
        hold_and_step(q)
        a_u, a_g, _ = geom()
        worse = max(a_u, a_g)
        flag = ""
        if best is None or worse < best[1]:
            best = (v, worse, a_u, a_g); flag = "  <-- best so far"
        print(f"  {v:11.4f} {math.degrees(v):7.1f} {a_u:9.1f} {a_g:9.1f} {worse:13.1f}{flag}", flush=True)
    print(f"  BEST {label} = {best[0]:.4f} rad ({math.degrees(best[0]):.1f} deg)  "
          f"-> UR5e {best[2]:.1f} deg, GEN3 {best[3]:.1f} deg", flush=True)
    return best

base_worse = max(a_u, a_g)
results = {}
for name in [x.strip() for x in args_cli.joints.split(',') if x.strip()]:
    col = arm_names.index(name)
    results[name] = sweep(col, name)

def sweep_pair(colA, nameA, colB, nameB, n=8):
    print(f"\n=== JOINT SWEEP {nameA} x {nameB} ({n}x{n} grid) ===", flush=True)
    best = None
    for ia in range(n):
        va = -math.pi + 2 * math.pi * ia / n
        row = []
        for ib in range(n):
            vb = -math.pi + 2 * math.pi * ib / n
            q = {idx: q0[idx].clone() for idx in (U, G3)}
            for idx in (U, G3):
                q[idx][:, colA] = va
                q[idx][:, colB] = vb
            hold_and_step(q)
            au, ag, _ = geom()
            worse = max(au, ag)
            row.append(worse)
            if best is None or worse < best[2]:
                best = (va, vb, worse, au, ag)
        print(f"  {nameA}={va:7.3f}  " + " ".join(f"{w:5.1f}" for w in row), flush=True)
    print(f"  BEST {nameA}={best[0]:.4f} {nameB}={best[1]:.4f} -> worse {best[2]:.1f} deg "
          f"(UR5e {best[3]:.1f}, GEN3 {best[4]:.1f})", flush=True)
    return best

pair_best = {}
for spec in [x.strip() for x in args_cli.pairs.split(',') if x.strip()]:
    nA, nB = spec.split(':')
    pair_best[spec] = sweep_pair(arm_names.index(nA), nA, arm_names.index(nB), nB)

print(f"\n=== SUMMARY ===", flush=True)
print(f"  baseline worse-of-two = {base_worse:.1f} deg", flush=True)
for name, b in results.items():
    print(f"  best via {name:14s} = {b[1]:6.1f} deg at {b[0]:8.4f} rad "
          f"(config now {float(q0[U][0, arm_names.index(name)]):.4f})", flush=True)
for spec, b in pair_best.items():
    print(f"  best via {spec:14s} = {b[2]:6.1f} deg at {b[0]:.4f} / {b[1]:.4f} rad", flush=True)
print(f"  TARGET: both angles near 0 deg means palms point at each other.", flush=True)
print("DIAG_PALM_FACING_DONE", flush=True)
simulation_app.close()
