"""Paired held-out arm-start evaluation for bimanual handover checkpoints."""

import argparse
import json
import os
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--max-steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(sys.argv[1:] + ["--experience", "/isaac-sim/apps/omni.isaac.sim.python.kit"])
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa: F401

sys.path.insert(
    0,
    "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train",
)
import networks  # noqa: F401

os.makedirs(args.output, exist_ok=True)
cfg = parse_env_cfg("Isaac-Bimanual-Bidirectional-v0", device=args.device, num_envs=1)
cfg.randomize_giver = True
cfg.use_reverse_curriculum = True
cfg.arm_start_randomization = True
cfg.arm_start_nominal_fraction = 0.0
# Held out from training: twice the shoulder and elbow perturbation spans.
cfg.arm_start_shoulder_range_rad = 0.06
cfg.arm_start_elbow_range_rad = 0.05
env = gym.make("Isaac-Bimanual-Bidirectional-v0", cfg=cfg)
u = env.unwrapped
u.curriculum_level = 4
u._update_curriculum = lambda success, done: None
model = PPO.load(args.checkpoint, device="cpu")
model.policy.eval()
arms = [u.scene.articulations[key] for key in u.cfg.keys]
object_body = u.scene.rigid_objects["object"]
rows = []

for episode in range(args.episodes):
    desired_giver = episode % 2
    seed = 10000 + episode
    torch.manual_seed(seed)
    np.random.seed(seed)
    for attempt in range(100):
        obs, _ = env.reset()
        if int(u.giver_is_gen3[0]) == desired_giver:
            break
    else:
        raise RuntimeError(f"Could not sample giver role {desired_giver}")

    initial_arm_q = [
        art.data.joint_pos[0, u._robot_joints_idx[i]].cpu().tolist()
        for i, art in enumerate(arms)
    ]
    initial_object_xyz = object_body.data.root_state_w[0, :3].cpu().tolist()
    stages = []
    finished = False
    timed_out_final = False
    nonfinite = False
    for step in range(args.max_steps):
        actions, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
        obs, _, terminated, timed_out, _ = env.step(torch.as_tensor(actions, device=u.device))
        stages.append(int(u.stage[0]))
        if not torch.isfinite(object_body.data.root_state_w[0]).all():
            nonfinite = True
            break
        if bool(terminated[0] or timed_out[0]):
            finished = True
            timed_out_final = bool(timed_out[0])
            break
    row = {
        "episode": episode + 1,
        "seed": seed,
        "giver_slot": desired_giver,
        "reset_attempts": attempt + 1,
        "initial_arm_q_rad": initial_arm_q,
        "initial_object_world_xyz_m": initial_object_xyz,
        "steps": len(stages),
        "peak_stage": max(stages, default=0),
        "finished": finished,
        "ended_by_timeout": timed_out_final,
        "nonfinite": nonfinite,
        "success": max(stages, default=0) == 4 and finished and not timed_out_final and not nonfinite,
    }
    rows.append(row)
    if episode % 16 == 15:
        print("EVAL_PROGRESS", episode + 1, "success", sum(x["success"] for x in rows), flush=True)

with open(os.path.join(args.output, "episodes.json"), "w") as handle:
    json.dump({"checkpoint": args.checkpoint, "config": {
        "shoulder_range_rad": cfg.arm_start_shoulder_range_rad,
        "elbow_range_rad": cfg.arm_start_elbow_range_rad,
        "nominal_fraction": cfg.arm_start_nominal_fraction,
        "deterministic_policy": True,
    }, "episodes": rows}, handle, indent=2)
print("RANDOMIZED_ARM_EVAL_DONE", sum(x["success"] for x in rows), len(rows), flush=True)
env.close()
simulation_app.close()
