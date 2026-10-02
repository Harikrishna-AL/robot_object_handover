"""
Which way do the two palms actually face, and what wrist angle makes them face
each other?

Prompted by an observation on the 409M video: "the back of their hands are facing
each other, that makes the task harder". Both slots hold identical GEN3 hardware
with identical arm joint angles, and robot_cfg_1 is rotated 180 deg about z
(rot=(0,0,0,1)), so the two palm normals are related by that same rotation --
meaning if one faces away from the other arm, BOTH do. That is consistent with
the observation and with the measured failure: in the UR5e-gives direction the
object arrives 0.095m from the receiver's fingertips and a grasp never closes.

"Palm direction" is measured WITHOUT relying on USD axis conventions:

    grasp_dir = normalize(mean_fingertip_pos - palm_pos)

i.e. the direction the fingers extend and curl toward. Palms face each other when
each arm's grasp_dir points at the other arm's palm. Reported as the angle
between grasp_dir and the vector to the other palm: 0 deg = facing it, 180 deg =
facing away.

Then sweeps arm_joint_7 (the final wrist rotation) over a full circle and reports
the angle at each value, for both arms, so the config can be set to a measured
value rather than a guessed one. If joint_7 alone cannot fix it, arm_joint_5 is
swept too -- the palm normal may not be perpendicular to the tool axis, in which
case rolling the tool does not reorient the palm at all.

Joint state is written and then the SIM IS STEPPED before reading poses:
write_joint_state_to_sim refreshes joint_pos but NOT body_state_w (link
transforms need a real physics step), which is the trap that made an earlier IK
placement attempt diverge (job 4166925).
"""
import argparse, math, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--settle', type=int, default=30, help='sim steps to settle after each write')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, torch
from omni.isaac.lab_tasks.utils import parse_env_cfg
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
        f = art.data.body_state_w[:, u.finger_tips[idx], 0:3].mean(1)
        pos[idx] = p
        gdir[idx] = torch.nn.functional.normalize(f - p, dim=-1)
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
print(f"  UR5e-slot (screen-LEFT)  grasp_dir vs toward-other = {a_u:6.1f} deg", flush=True)
print(f"  GEN3-slot (screen-RIGHT) grasp_dir vs toward-other = {a_g:6.1f} deg", flush=True)
print(f"  (0 = palm/fingers point at the other arm, 180 = point away)", flush=True)

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

b7 = sweep(j7, 'arm_joint_7')
b5 = sweep(j5, 'arm_joint_5')

print(f"\n=== SUMMARY ===", flush=True)
print(f"  baseline worse-of-two      = {max(a_u, a_g):.1f} deg", flush=True)
print(f"  best via arm_joint_7       = {b7[1]:.1f} deg at {b7[0]:.4f} rad", flush=True)
print(f"  best via arm_joint_5       = {b5[1]:.1f} deg at {b5[0]:.4f} rad", flush=True)
print(f"  current config arm_joint_7 = {float(q0[U][0, j7]):.4f} rad, arm_joint_5 = {float(q0[U][0, j5]):.4f} rad", flush=True)
print("DIAG_PALM_FACING_DONE", flush=True)
simulation_app.close()
