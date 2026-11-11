"""Render eight level-4 handovers, alternating the giving role."""

import argparse
import json
import os
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(sys.argv[1:] + ["--experience", "/isaac-sim/apps/omni.isaac.sim.python.kit"])
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
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
env = gym.make("Isaac-Bimanual-Bidirectional-v0", cfg=cfg, render_mode="rgb_array")
u = env.unwrapped
u.curriculum_level = 4
model = PPO.load(args.checkpoint, device="cpu")
model.policy.eval()
object_body = u.scene.rigid_objects["object"]
arms = [u.scene.articulations[key] for key in u.cfg.keys]
palm_ids = [arms[i].find_bodies("hand_palm_link")[0][0] for i in range(2)]
rows = []

for episode in range(args.episodes):
    desired_giver = episode % 2
    for _ in range(100):
        obs, _ = env.reset()
        if int(u.giver_is_gen3[0]) == desired_giver:
            break
    else:
        raise RuntimeError(f"Could not sample giver role {desired_giver}")

    center = torch.stack([arms[i].data.body_state_w[0, palm_ids[i], :3] for i in range(2)]).mean(0)
    center = center.cpu().numpy()
    eye = center + np.array([0.65, 1.5, 0.65])
    target = center + np.array([0.0, 0.0, -0.18])
    u.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
    name = f"episode_{episode + 1:02d}_giver_slot{desired_giver}"
    path = os.path.join(args.output, name + ".mp4")
    writer = imageio.get_writer(path, fps=1 / u.step_dt, codec="libx264", quality=8, macro_block_size=1)
    stages = []
    receiver_grasped = []
    giver_grasped = []
    object_speeds = []
    finished = False
    for step in range(args.steps):
        actions, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
        obs, _, terminated, timed_out, _ = env.step(torch.as_tensor(actions, device=u.device))
        stages.append(int(u.stage[0]))
        if bool(terminated[0] or timed_out[0]):
            finished = True
            break
        contact_matrix = u.cfg.contact_matrix[u._ur5e_contact_slice]
        grasps = [bool(u._grasp_flags(i, contact_matrix)[1][0]) for i in range(2)]
        giver_grasped.append(grasps[desired_giver])
        receiver_grasped.append(grasps[1 - desired_giver])
        object_speeds.append(float(torch.norm(object_body.data.root_state_w[0, 7:10])))
        u.sim.render()
        frame = u.render()
        if step == 10 and not np.any(frame):
            raise RuntimeError("Blank RGB frames")
        writer.append_data(frame)
    writer.close()
    row = {
        "video": os.path.basename(path),
        "episode": episode + 1,
        "giver_slot": desired_giver,
        "receiver_slot": 1 - desired_giver,
        "curriculum_level": 4,
        "deterministic": True,
        "steps": len(stages),
        "finished": finished,
        "peak_stage": max(stages),
        "last_10_giver_grasp_fraction": sum(giver_grasped[-10:]) / max(1, min(10, len(giver_grasped))),
        "last_10_receiver_grasp_fraction": sum(receiver_grasped[-10:]) / max(1, min(10, len(receiver_grasped))),
        "max_object_speed": max(object_speeds, default=0.0),
        "stages": stages,
    }
    rows.append(row)
    print("MULTI_EPISODE", json.dumps({k: v for k, v in row.items() if k != "stages"}), flush=True)

with open(os.path.join(args.output, "episodes.json"), "w") as handle:
    json.dump({"checkpoint": args.checkpoint, "episodes": rows}, handle, indent=2)
print("MULTI_EPISODE_RENDER_DONE", flush=True)
env.close()
simulation_app.close()
