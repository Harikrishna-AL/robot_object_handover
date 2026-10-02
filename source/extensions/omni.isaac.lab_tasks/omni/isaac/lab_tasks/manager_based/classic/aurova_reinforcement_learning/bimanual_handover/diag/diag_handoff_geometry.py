"""
Is the working handover direction actually being SOLVED, or is it geometrically
trivial?

Prompted by an observation on the 409M role-swap video: in the direction that
works (GEN3 gives, UR5e-slot arm receives -- screen-right gives, screen-left
takes), the receiving hand looks like it is already on the object and takes it
"without changing much of hand's position", suggesting hand/object interference
rather than a real grasp. The numbers are consistent with that reading:
successful episodes last 91.7 steps against 391.8 for failures, and the stage
histogram jumps almost straight to success (s2:17 s3:45 s4:488).

If true, 88.8% in that direction is not dexterity -- it is a near-degenerate
grasp, and "role randomisation helped" would partly mean "it found the direction
where the task is trivial". That would change what the whole experiment means, so
it needs measuring.

Reported PER ROLE, at reset and at the moment each stage is first reached:
  - receiver fingertips -> object distance
  - receiver end-effector distance travelled since reset (does the hand move?)
  - summed contact force on the receiver's hand (a spike implies penetration)
  - handoff point distance to each arm's end effector (is the meet point
    asymmetrically placed between the two arms?)

A trivial direction looks like: small tips->obj at reset, near-zero travel to
grasp, and a large force spike. A solved direction looks like: large tips->obj at
reset, substantial travel, moderate force.
"""
import argparse, os, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--steps', type=int, default=2500)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym, torch
from stable_baselines3 import PPO
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
import networks  # noqa: F401

env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped
model = PPO.load(args_cli.checkpoint)
print(f"LOADED {args_cli.checkpoint}", flush=True)
print(f"randomize_giver = {u.cfg.randomize_giver}   num_envs = {u.num_envs}", flush=True)

U, G3 = u.cfg.UR5e, u.cfg.GEN3
dev = u.device

def tips(idx):
    return u.scene.articulations[u.cfg.keys[idx]].data.body_state_w[:, u.finger_tips[idx], 0:3].mean(1)

def ee(idx):
    return u.scene.articulations[u.cfg.keys[idx]].data.body_state_w[:, u.ee_jacobi_idx[idx] + 1, 0:3]

def obj():
    return u.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]

def hand_force(idx):
    sl = u._gen3_contact_slice if idx == G3 else u._ur5e_contact_slice
    return u.force_norms[:, sl].sum(-1)

def by_role(v_u, v_g):
    """(giver, receiver) per env, matching the env's own convention."""
    m = u.giver_is_gen3
    while m.dim() < v_u.dim():
        m = m.unsqueeze(-1)
    return torch.where(m, v_g, v_u), torch.where(m, v_u, v_g)

# accumulators: [role][stage] -> list
import collections
acc = {0: collections.defaultdict(list), 1: collections.defaultdict(list)}
reset_stats = {0: collections.defaultdict(list), 1: collections.defaultdict(list)}

obs, _ = env.reset()
role = u.giver_is_gen3.clone()
prev_stage = u.stage.clone()
_, recv_ee0 = by_role(ee(U), ee(G3))
recv_ee_prev = recv_ee0.clone()
travel = torch.zeros(u.num_envs, device=dev)

def record_reset(mask):
    """Geometry at the start of an episode, per role."""
    if not torch.any(mask):
        return
    _, d_recv = by_role(torch.norm(tips(U) - obj(), dim=-1), torch.norm(tips(G3) - obj(), dim=-1))
    d_give, _ = by_role(torch.norm(tips(U) - obj(), dim=-1), torch.norm(tips(G3) - obj(), dim=-1))
    h_u = torch.norm(u.handoff_pos - ee(U), dim=-1)
    h_g = torch.norm(u.handoff_pos - ee(G3), dim=-1)
    h_give, h_recv = by_role(h_u, h_g)
    for i in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
        r = int(role[i].item())
        reset_stats[r]['recv_tips_to_obj'].append(float(d_recv[i]))
        reset_stats[r]['give_tips_to_obj'].append(float(d_give[i]))
        reset_stats[r]['handoff_to_receiver_ee'].append(float(h_recv[i]))
        reset_stats[r]['handoff_to_giver_ee'].append(float(h_give[i]))

record_reset(torch.ones(u.num_envs, dtype=torch.bool, device=dev))

for step in range(args_cli.steps):
    a_np, _ = model.predict(obs['policy'].cpu().numpy(), deterministic=True)
    obs, rew, fell, ended, info = env.step(torch.as_tensor(a_np, device=dev))

    _, recv_ee_now = by_role(ee(U), ee(G3))
    travel = travel + torch.norm(recv_ee_now - recv_ee_prev, dim=-1)
    recv_ee_prev = recv_ee_now

    # On each FIRST entry to a stage, capture what the receiver's hand was doing.
    newly = u.stage > prev_stage
    if torch.any(newly):
        _, d_recv = by_role(torch.norm(tips(U) - obj(), dim=-1), torch.norm(tips(G3) - obj(), dim=-1))
        _, f_recv = by_role(hand_force(U), hand_force(G3))
        for i in newly.nonzero(as_tuple=False).squeeze(-1).tolist():
            r = int(role[i].item()); st = int(u.stage[i].item())
            acc[r][f's{st}_tips_to_obj'].append(float(d_recv[i]))
            acc[r][f's{st}_recv_travel'].append(float(travel[i]))
            acc[r][f's{st}_recv_force'].append(float(f_recv[i]))
    prev_stage = u.stage.clone()

    done = torch.logical_or(fell, ended)
    if torch.any(done):
        idx = done.nonzero(as_tuple=False).squeeze(-1)
        role[idx] = u.giver_is_gen3[idx]
        prev_stage[idx] = u.stage[idx]
        travel[idx] = 0.0
        recv_ee_prev[idx] = recv_ee_now[idx]
        record_reset(done)

def show(label, d, keys):
    print(f"\n--- {label} ---", flush=True)
    for k in keys:
        v = d.get(k, [])
        if not v:
            print(f"  {k:26s}  (never)", flush=True); continue
        v_sorted = sorted(v)
        med = v_sorted[len(v_sorted) // 2]
        print(f"  {k:26s}  n={len(v):4d}  mean={sum(v)/len(v):7.4f}  median={med:7.4f}  "
              f"min={v_sorted[0]:7.4f}  max={v_sorted[-1]:7.4f}", flush=True)

for r, name in ((0, 'UR5e gives  (screen-LEFT gives, right receives)  -- the 0% direction'),
                (1, 'GEN3 gives  (screen-RIGHT gives, left receives)  -- the 88% direction')):
    show(f"{name}  AT RESET", reset_stats[r],
         ['recv_tips_to_obj', 'give_tips_to_obj', 'handoff_to_receiver_ee', 'handoff_to_giver_ee'])
    show(f"{name}  AT STAGE ENTRY", acc[r],
         ['s1_tips_to_obj', 's2_tips_to_obj', 's2_recv_travel', 's2_recv_force',
          's3_tips_to_obj', 's3_recv_travel', 's3_recv_force',
          's4_tips_to_obj', 's4_recv_travel', 's4_recv_force'])

print("\nDIAG_HANDOFF_GEOMETRY_DONE", flush=True)
simulation_app.close()
