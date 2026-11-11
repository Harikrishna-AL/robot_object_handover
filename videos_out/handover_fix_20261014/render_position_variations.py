"""Evaluate one checkpoint on controlled arm/object reset variations."""

import argparse
import json
import os
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
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
u._update_curriculum = lambda success, done: None
model = PPO.load(args.checkpoint, device="cpu")
model.policy.eval()
object_body = u.scene.rigid_objects["object"]
arms = [u.scene.articulations[key] for key in u.cfg.keys]
palm_ids = [arms[i].find_bodies("hand_palm_link")[0][0] for i in range(2)]
rows = []
base_arm_q = np.asarray(u.cfg.reset_arm_joint_pos, dtype=float).copy()
base_obj_offset = u.cfg.obj_pos_trans[0].clone()

# Small offsets that keep the bar inside the giver's physical grip. These are
# controlled out-of-training-distribution tests, not new training examples.
cases = [
    ("baseline", None, None, (0.0, 0.0, 0.0)),
    ("giver_shoulder_plus_0p06rad", "giver", (0, 0.06), (0.0, 0.0, 0.0)),
    ("receiver_shoulder_minus_0p06rad", "receiver", (0, -0.06), (0.0, 0.0, 0.0)),
    ("both_elbows_opposite_0p05rad", "both", (2, 0.05), (0.0, 0.0, 0.0)),
    ("object_local_x_plus_6mm", None, None, (0.006, 0.0, 0.0)),
    ("object_local_x_minus_6mm", None, None, (-0.006, 0.0, 0.0)),
    ("object_local_z_plus_6mm", None, None, (0.0, 0.0, 0.006)),
    ("object_local_z_minus_6mm", None, None, (0.0, 0.0, -0.006)),
]

for episode in range(len(cases) * 2):
    case_name, arm_target, joint_change, object_delta = cases[episode // 2]
    desired_giver = episode % 2
    arm_q = base_arm_q.copy()
    if arm_target == "giver":
        arm_q[desired_giver, joint_change[0]] += joint_change[1]
    elif arm_target == "receiver":
        arm_q[1 - desired_giver, joint_change[0]] += joint_change[1]
    elif arm_target == "both":
        arm_q[desired_giver, joint_change[0]] += joint_change[1]
        arm_q[1 - desired_giver, joint_change[0]] -= joint_change[1]
    u.cfg.reset_arm_joint_pos = arm_q.tolist()
    u.cfg.obj_pos_trans[0] = base_obj_offset + torch.as_tensor(object_delta, device=u.device)
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
    palm_xyz = [arms[i].data.body_state_w[0, palm_ids[i], :3].cpu().tolist() for i in range(2)]
    object_xyz = object_body.data.root_state_w[0, :3].cpu().tolist()
    name = f"episode_{episode + 1:02d}_{case_name}_giver_slot{desired_giver}"
    path = os.path.join(args.output, name + ".mp4")
    writer = imageio.get_writer(path, fps=1 / u.step_dt, codec="libx264", quality=8, macro_block_size=1)
    stages = []
    receiver_grasped = []
    giver_grasped = []
    object_speeds = []
    finished = False
    ended_by_timeout = False
    for step in range(args.steps):
        actions, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
        obs, _, terminated, timed_out, _ = env.step(torch.as_tensor(actions, device=u.device))
        stages.append(int(u.stage[0]))
        if bool(terminated[0] or timed_out[0]):
            finished = True
            ended_by_timeout = bool(timed_out[0])
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
        "case": case_name,
        "arm_joint_delta_rad": (float(joint_change[1]) if joint_change else 0.0),
        "object_local_delta_m": object_delta,
        "initial_palm_world_xyz_m": palm_xyz,
        "initial_object_world_xyz_m": object_xyz,
        "giver_slot": desired_giver,
        "receiver_slot": 1 - desired_giver,
        "curriculum_level": 4,
        "deterministic": True,
        "steps": len(stages),
        "finished": finished,
        "ended_by_timeout": ended_by_timeout,
        "peak_stage": max(stages),
        "success": max(stages) == 4 and finished and not ended_by_timeout,
        "last_10_giver_grasp_fraction": sum(giver_grasped[-10:]) / max(1, min(10, len(giver_grasped))),
        "last_10_receiver_grasp_fraction": sum(receiver_grasped[-10:]) / max(1, min(10, len(receiver_grasped))),
        "max_object_speed": max(object_speeds, default=0.0),
        "stages": stages,
    }
    rows.append(row)
    print("MULTI_EPISODE", json.dumps({k: v for k, v in row.items() if k != "stages"}), flush=True)

with open(os.path.join(args.output, "episodes.json"), "w") as handle:
    json.dump({"checkpoint": args.checkpoint, "episodes": rows}, handle, indent=2)
print("POSITION_VARIATION_RENDER_DONE", flush=True)
env.close()
simulation_app.close()
