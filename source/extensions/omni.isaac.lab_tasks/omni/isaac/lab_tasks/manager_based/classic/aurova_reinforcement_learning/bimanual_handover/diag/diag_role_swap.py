"""
Does per-episode role randomisation actually work, in BOTH directions?

Four things have to be true, and each one has failed before in some form:

  1. BALANCE   -- giver_is_gen3 is ~50/50 across episodes. If it is stuck at 0
                  the flag is a no-op and the context scalar is still useless.
  2. GRIP      -- the object is HELD in the giver's hand for both roles, with
                  the weld off. robot_cfg_2 (the GEN3 slot) does NOT bake
                  _CLOSED_HAND_JOINTS into its init_state, so if _assign_roles
                  fails to write the hand state the GEN3-gives half starts with
                  the object in an open palm and it free-falls (measured 242mm
                  in 111 steps before the closed-grip fix).
  3. SCALAR    -- the +/-1 in the observation agrees with who is actually
                  giving. This is the whole point: if the scalar and the role
                  disagree, the policy is being lied to.
  4. PLACEMENT -- the object starts inside the GIVER's fingers, not the other
                  arm's. The parent _reset_idx always puts it at UR5e's end
                  effector, so this catches a missing re-placement directly.

Both arms are commanded still throughout, so anything that happens to the
object is physics, not policy -- the same zero-action control that caught the
sinking-object bug.

PASS  balance ~0.5, both groups HELD, scalar matches, giver closer than receiver
"""
import argparse, sys
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--steps', type=int, default=300)
parser.add_argument('--resets', type=int, default=20, help='extra resets sampled for the balance check')
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
env_cfg.randomize_giver = True
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped

U, G3 = u.cfg.UR5e, u.cfg.GEN3
art = {U: u.scene.articulations[u.cfg.keys[U]], G3: u.scene.articulations[u.cfg.keys[G3]]}
D = u.cfg.arm_obs_dim                      # role scalar is the LAST column of each block
print(f"weld = {args_cli.weld}   randomize_giver = {env_cfg.randomize_giver}   "
      f"num_envs = {args_cli.num_envs}   arm_obs_dim = {D}", flush=True)

obs, _ = env.reset()
policy = obs["policy"] if isinstance(obs, dict) else obs

# ---- 1. BALANCE ---------------------------------------------------------
# Sampled over many resets, not one -- a single reset of N envs is one draw of
# the same Bernoulli and says nothing about the reset path being re-run.
seen, hits = 0, 0
for _ in range(args_cli.resets):
    env.reset()
    seen += u.num_envs
    hits += int(u.giver_is_gen3.sum().item())
print(f"\n[1] BALANCE   gen3-gives fraction = {hits/max(seen,1):.3f}  "
      f"({hits}/{seen} env-episodes)", flush=True)

obs, _ = env.reset()
policy = obs["policy"] if isinstance(obs, dict) else obs
m = u.giver_is_gen3.clone()
n_gen3, n_ur5e = int(m.sum().item()), int((~m).sum().item())
print(f"    this episode: {n_ur5e} envs UR5e-gives, {n_gen3} envs GEN3-gives", flush=True)

# ---- 3. SCALAR ----------------------------------------------------------
role_ur5e, role_gen3 = policy[:, D - 1], policy[:, 2 * D - 1]
# A giver must read -1 ("you hold it, give it away"), a receiver +1.
want_ur5e = torch.where(m, torch.ones_like(role_ur5e), -torch.ones_like(role_ur5e))
want_gen3 = -want_ur5e
ok_scalar = torch.equal(role_ur5e, want_ur5e) and torch.equal(role_gen3, want_gen3)
print(f"[3] SCALAR    role_ur5e unique = {sorted(set(role_ur5e.tolist()))}  "
      f"role_gen3 unique = {sorted(set(role_gen3.tolist()))}  "
      f"matches role: {'YES' if ok_scalar else 'NO'}", flush=True)
if not ok_scalar:
    bad = (role_ur5e != want_ur5e).nonzero().flatten()[:5].tolist()
    print(f"    mismatched envs (first 5): {bad}", flush=True)

def tip_dist(idx):
    tips = art[idx].data.body_state_w[:, u.finger_tips[idx], 0:3].mean(1)
    obj = u.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]
    return torch.norm(tips - obj, dim=-1)

# ---- 4. PLACEMENT -------------------------------------------------------
d_u, d_g = tip_dist(U), tip_dist(G3)
d_giver = torch.where(m, d_g, d_u)
d_recv = torch.where(m, d_u, d_g)
ok_place = bool((d_giver < d_recv).all().item())
print(f"[4] PLACEMENT giver tips->obj = {d_giver.mean().item():.4f} m   "
      f"receiver tips->obj = {d_recv.mean().item():.4f} m   "
      f"object in giver's hand: {'YES' if ok_place else 'NO'} "
      f"({int((d_giver < d_recv).sum().item())}/{u.num_envs} envs)", flush=True)

# ---- 2. GRIP ------------------------------------------------------------
zero = torch.zeros((args_cli.num_envs, env.action_space.shape[-1]))
obj0 = u.scene.rigid_objects["object"].data.body_state_w[:, 0, 2].clone()
print(f"\n{'step':>5} {'obj_z(ur5e)':>12} {'obj_z(gen3)':>12} "
      f"{'Fgiver(ur5e)':>13} {'Fgiver(gen3)':>13}", flush=True)
for step in range(args_cli.steps):
    obs, r, te, tr, info = env.step(zero)
    if step < 3 or step % 60 == 0 or step == args_cli.steps - 1:
        z = u.scene.rigid_objects["object"].data.body_state_w[:, 0, 2]
        f_u = u.contacts[:, u._ur5e_contact_slice].sum(-1)
        f_g = u.contacts[:, u._gen3_contact_slice].sum(-1)
        zu = z[~m].mean().item() if n_ur5e else float('nan')
        zg = z[m].mean().item() if n_gen3 else float('nan')
        fu = f_u[~m].mean().item() if n_ur5e else float('nan')
        fg = f_g[m].mean().item() if n_gen3 else float('nan')
        print(f"{step:>5} {zu:>12.3f} {zg:>12.3f} {fu:>13.3f} {fg:>13.3f}", flush=True)

zf = u.scene.rigid_objects["object"].data.body_state_w[:, 0, 2]
drop = obj0 - zf
drop_u = drop[~m].mean().item() if n_ur5e else 0.0
drop_g = drop[m].mean().item() if n_gen3 else 0.0
print(f"\n[2] GRIP      UR5e-gives drop = {drop_u:+.4f} m   "
      f"GEN3-gives drop = {drop_g:+.4f} m   over {args_cli.steps} steps", flush=True)
held_u, held_g = abs(drop_u) < 0.02, abs(drop_g) < 0.02
print(f"    UR5e-gives: {'HELD' if held_u else 'SLIPPED'}   "
      f"GEN3-gives: {'HELD' if held_g else 'SLIPPED'}", flush=True)

balance_ok = 0.35 <= hits / max(seen, 1) <= 0.65
print(f"\nVERDICT: balance={'ok' if balance_ok else 'FAIL'} "
      f"grip={'ok' if (held_u and held_g) else 'FAIL'} "
      f"scalar={'ok' if ok_scalar else 'FAIL'} "
      f"placement={'ok' if ok_place else 'FAIL'}", flush=True)
print("DIAG_ROLE_SWAP_DONE", flush=True)
simulation_app.close()
