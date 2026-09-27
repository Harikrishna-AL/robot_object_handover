"""
Does the shared policy do BOTH roles, or only one?

Runs a trained checkpoint and counts completed episodes separately for
"UR5e gives" and "GEN3 gives". This is the question the +/-1 context scalar
exists to answer, and no aggregate metric can settle it.

Why it is needed: metrics/giver_is_gen3_frac sits at ~0.33 in run 4295173 even
though the reset draw is 50/50 (diagnostic 4293464 measured 0.519 over 640
resets). That metric is sampled at episode-end instants, so ~1/3 means
GEN3-gives episodes are about HALF as long as UR5e-gives ones -- which is either
"succeeds faster" (fine) or "fails fast" (the shared policy learned one role
only). Opposite conclusions, same number.

Counted per role, per episode:
  peak stage reached (0..4), success (stage 4, untruncated), truncation cause,
  and episode length.

Episodes are attributed by the role each env had AT RESET, latched for the
episode -- not read at termination, which is exactly the bias that made the
aggregate ambiguous.
"""
import argparse, sys
from collections import defaultdict
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Isaac-Bimanual-Bidirectional-v0')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--steps', type=int, default=4000)
# For checkpoints trained BEFORE randomize_giver existed (e.g. the fixed-role
# control 4292718): evaluating them with randomisation on puts the object in
# the other arm's hand half the time, a situation they never saw in training.
# Evaluate with sampled rather than mean actions, i.e. the policy AS TRAINED.
# Matters when a policy's competence depends on its action noise: the control
# 4292718 scored 3.5% deterministically while its training logs showed
# stage1_fraction 0.647 and episodes ~180 steps vs 343 deterministic.
parser.add_argument('--stochastic', action='store_true',
                    help='sample actions (deterministic=False), as during training')
parser.add_argument('--fixed_roles', action='store_true',
                    help='force randomize_giver=False, as pre-randomisation checkpoints were trained')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(sys.argv[1:] + ['--experience', '/isaac-sim/apps/omni.isaac.sim.python.kit'])
args_cli.headless = True
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import os
import gymnasium as gym, torch
from stable_baselines3 import PPO
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa

# The checkpoint cloudpickles SharedArmActorCriticPolicy, so the module that
# defines it must be importable under the SAME name it had at save time.
# train.py does a bare `from networks import *` and gets away with it because
# Python puts a script's own directory on sys.path -- networks.py sits next to
# train.py. This script lives in diag/, so without this PPO.load dies with
# "ModuleNotFoundError: No module named 'networks'" (job 4307343).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))
import networks  # noqa: F401  (registers SharedArmActorCriticPolicy for unpickling)

env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
if args_cli.fixed_roles:
    env_cfg.randomize_giver = False
    print('FIXED_ROLES: randomize_giver=False', flush=True)
env = gym.make(args_cli.task, cfg=env_cfg); u = env.unwrapped
model = PPO.load(args_cli.checkpoint)
print(f"LOADED {args_cli.checkpoint}", flush=True)
print(f"randomize_giver = {u.cfg.randomize_giver}   num_envs = {u.num_envs}   "
      f"action_selection = {'stochastic' if args_cli.stochastic else 'deterministic'}", flush=True)

obs, _ = env.reset()
N = u.num_envs
# Role latched at reset; peak stage tracked within the episode. Both must be
# read BEFORE the env resets the env, so they are updated every step.
role = u.giver_is_gen3.clone()
peak = u.stage.clone()
ep_start = torch.zeros(N, dtype=torch.long, device=u.device)
step_i = 0

stats = {0: defaultdict(int), 1: defaultdict(int)}   # 0 = UR5e gives, 1 = GEN3 gives
lens = {0: [], 1: []}
peaks = {0: defaultdict(int), 1: defaultdict(int)}

for step_i in range(args_cli.steps):
    a_np, _ = model.predict(obs['policy'].cpu().numpy(), deterministic=not args_cli.stochastic)
    # CAREFUL with these two. IsaacLab requires _get_dones() to return
    # (terminated, time_out) -- direct_rl_env.py:338 unpacks it as
    # `self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()`.
    # This env family returns (truncated, terminated) instead, swapped, since the
    # baseline (bimanual_direct_env.py:826, commit 4a60502, June 2026) and
    # inherited by the bidirectional and ICL envs. So:
    #   fell  = the env's OWN `truncated`  -> falling / out-of-bounds / ground
    #   ended = the env's OWN `terminated` -> time_out OR transfer_complete
    # An earlier version of this script called the second one `trunc` and scored
    # `peak >= 4 and not trunc`, which is unsatisfiable: reaching stage 4 sets
    # transfer_complete, which sets that very flag. Job 4307499 therefore
    # reported success=0 while its own histogram showed 260 episodes peaking at
    # stage 4. Success is simply peak stage 4 -- the ladder is a one-way ratchet
    # and stage 4 already requires the receiver to still hold the object after
    # the giver let go.
    obs, rew, fell, ended, info = env.step(torch.as_tensor(a_np, device=u.device))
    peak = torch.maximum(peak, u.stage)
    done = torch.logical_or(fell, ended)
    if torch.any(done):
        idx = done.nonzero(as_tuple=False).squeeze(-1)
        for i in idx.tolist():
            r = int(role[i].item())
            p = int(peak[i].item())
            stats[r]['episodes'] += 1
            peaks[r][p] += 1
            if p >= 4:
                stats[r]['success'] += 1
            if bool(fell[i].item()):
                stats[r]['fell'] += 1
            elif p < 4:
                stats[r]['timeout'] += 1
            lens[r].append(step_i - int(ep_start[i].item()))
        # re-latch for the fresh episodes
        role[idx] = u.giver_is_gen3[idx]
        peak[idx] = u.stage[idx]
        ep_start[idx] = step_i

print("\n================ PER-ROLE OUTCOMES ================", flush=True)
for r, name in ((0, 'UR5e gives'), (1, 'GEN3 gives')):
    n = stats[r]['episodes']
    if n == 0:
        print(f"{name:12s}  no completed episodes", flush=True); continue
    s = stats[r]['success']
    f = stats[r]['fell']
    to = stats[r]['timeout']
    ml = sum(lens[r]) / max(len(lens[r]), 1)
    print(f"{name:12s}  episodes={n:4d}  success={s:4d} ({100.0*s/n:5.1f}%)  "
          f"dropped/fell={f:4d} ({100.0*f/n:5.1f}%)  timeout={to:4d} ({100.0*to/n:5.1f}%)  "
          f"mean_len={ml:6.1f}", flush=True)
    dist = "  ".join(f"s{k}:{peaks[r][k]}" for k in range(5))
    print(f"{'':12s}  peak stage -> {dist}", flush=True)

tot = stats[0]['episodes'] + stats[1]['episodes']
tots = stats[0]['success'] + stats[1]['success']
print(f"\nOVERALL       episodes={tot}  success={tots} ({100.0*tots/max(tot,1):.1f}%)", flush=True)
if stats[0]['episodes'] and stats[1]['episodes']:
    r0 = stats[0]['success'] / stats[0]['episodes']
    r1 = stats[1]['success'] / stats[1]['episodes']
    print(f"VERDICT: UR5e-gives {100*r0:.1f}%  vs  GEN3-gives {100*r1:.1f}%  -> "
          f"{'BOTH ROLES LEARNED' if min(r0, r1) > 0.25 * max(r0, r1) else 'ONE ROLE DOMINATES'}", flush=True)
print("EVAL_PER_ROLE_DONE", flush=True)
simulation_app.close()
