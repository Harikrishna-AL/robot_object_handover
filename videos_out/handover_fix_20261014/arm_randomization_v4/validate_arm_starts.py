"""Smoke-test randomized arm resets with the preserved handover policy."""

import argparse
import json
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(sys.argv[1:] + ["--experience", "/isaac-sim/apps/omni.isaac.sim.python.kit"])
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa: F401

sys.path.insert(0, "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train")
import networks  # noqa: F401

cfg = parse_env_cfg("Isaac-Bimanual-Bidirectional-v0", device=args.device, num_envs=args.num_envs)
cfg.arm_start_randomization = True
cfg.arm_start_nominal_fraction = 0.25
cfg.arm_start_shoulder_range_rad = 0.03
cfg.arm_start_elbow_range_rad = 0.025
cfg.use_reverse_curriculum = False
env = gym.make("Isaac-Bimanual-Bidirectional-v0", cfg=cfg)
u = env.unwrapped
u._set_curriculum_level(4)
model = PPO.load(args.checkpoint, device="cpu")
model.policy.eval()
obs, _ = env.reset()

def snapshot():
    result = []
    for idx, key in enumerate(u.cfg.keys):
        art = u.scene.articulations[key]
        actual = art.data.joint_pos[:, u._robot_joints_idx[idx]]
        nominal = torch.as_tensor(u.cfg.reset_arm_joint_pos[idx], device=u.device)
        delta = actual - nominal
        result.append({
            "shoulder_min": float(delta[:, 0].min()),
            "shoulder_max": float(delta[:, 0].max()),
            "elbow_min": float(delta[:, 2].min()),
            "elbow_max": float(delta[:, 2].max()),
        })
    return result

initial = snapshot()
done_by_role = [0, 0]
first_step_done = 0
nonfinite_steps = 0
for step in range(args.steps):
    role_before_step = u.giver_is_gen3.clone()
    actions, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
    obs, _, terminated, timed_out, info = env.step(torch.as_tensor(actions, device=u.device))
    done = terminated | timed_out
    # extras/log is measured before DirectRLEnv resets completed environments.
    log = info.get("log", {})
    for role in (0, 1):
        role_mask = (role_before_step if role == 1 else ~role_before_step)
        done_by_role[role] += int((done & role_mask).sum())
    if step == 0:
        first_step_done = int(done.sum())
    for key in u.cfg.keys:
        nonfinite_steps += int((~torch.isfinite(u.scene.articulations[key].data.root_state_w)).any(dim=-1).sum())
    if (step + 1) % 50 == 0:
        print("PILOT_PROGRESS", json.dumps({"step": step + 1, "done_by_role": done_by_role, "first_step_done": first_step_done, "trunc_any": log.get("metrics/trunc_any"), "trunc_object_escaped": log.get("metrics/trunc_object_escaped")}), flush=True)

print("PILOT_DONE", json.dumps({"num_envs": args.num_envs, "steps": args.steps, "initial_joint_delta_rad": initial, "first_step_done": first_step_done, "done_by_role": done_by_role, "nonfinite_steps": nonfinite_steps}), flush=True)
env.close()
simulation_app.close()
