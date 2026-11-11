"""Render the trained policy in both giving roles at curriculum level 0."""
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
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

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
model = PPO.load(args.checkpoint, device="cpu")
model.policy.eval()
object_body = u.scene.rigid_objects["object"]
arts = [u.scene.articulations[key] for key in u.cfg.keys]
palms = [arts[i].find_bodies("hand_palm_link")[0][0] for i in range(2)]
rows = []

for deterministic in (True, False):
    for role in (0, 1):
        for attempt in range(50):
            obs, _ = env.reset()
            if int(u.giver_is_gen3[0]) == role:
                break
        else:
            raise RuntimeError(f"Could not sample giving role {role}")
        center = torch.stack(
            [arts[i].data.body_state_w[0, palms[i], :3] for i in range(2)]
        ).mean(0).cpu().numpy()
        origin = u.scene.env_origins[0].clone()
        name = f"checkpoint_{'det' if deterministic else 'stoch'}_giver_slot{role}"
        video_path = os.path.join(args.output, name + ".mp4")
        writer = imageio.get_writer(video_path, fps=1 / u.step_dt, codec="libx264", quality=8, macro_block_size=1)
        max_obj_displacement = 0.0
        max_receiver_distance = 0.0
        max_object_speed = 0.0
        max_receiver_contact_force = 0.0
        max_giver_contact_force = 0.0
        peak_stage = 0
        stages = []
        distances = []
        object_positions = []
        finished = False
        for step in range(args.steps):
            actions, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=deterministic)
            obs, reward, fell, ended, info = env.step(torch.as_tensor(actions, device=u.device))
            stage = int(u.stage[0])
            peak_stage = max(stage, peak_stage)
            obj = object_body.data.root_state_w[0, :3]
            tips = arts[1 - role].data.body_state_w[0, u.finger_tips[1 - role], :3].mean(0)
            distance = float(torch.norm(obj - tips))
            displacement = float(torch.norm(obj - origin))
            max_object_speed = max(max_object_speed, float(torch.norm(object_body.data.root_state_w[0, 7:10])))
            max_receiver_contact_force = max(
                max_receiver_contact_force,
                float(u.force_norms[0, [u._ur5e_contact_slice, u._gen3_contact_slice][1 - role]].sum()),
            )
            max_giver_contact_force = max(
                max_giver_contact_force,
                float(u.force_norms[0, [u._ur5e_contact_slice, u._gen3_contact_slice][role]].sum()),
            )
            max_receiver_distance = max(max_receiver_distance, distance)
            max_obj_displacement = max(max_obj_displacement, displacement)
            stages.append(stage)
            distances.append(distance)
            object_positions.append((obj - origin).cpu().tolist())
            eye = center + np.array([0.65, 1.5, 0.65])
            target = center + np.array([0.0, 0.0, -0.18])
            u.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            u.sim.render()
            frame = u.render()
            if step == 20 and not np.any(frame):
                raise RuntimeError("Blank RGB frames")
            writer.append_data(frame)
            if step in (20, 100, 200, 300):
                imageio.imwrite(os.path.join(args.output, name + f"_{step}.png"), frame)
            if bool(fell[0] or ended[0]):
                finished = True
                break
        writer.close()
        row = dict(
            video=os.path.basename(video_path),
            giver_slot=role,
            deterministic=deterministic,
            steps=len(stages),
            finished=finished,
            peak_stage=peak_stage,
            max_receiver_distance=max_receiver_distance,
            max_object_distance_from_env_origin=max_obj_displacement,
            max_object_speed=max_object_speed,
            max_receiver_contact_force=max_receiver_contact_force,
            max_giver_contact_force=max_giver_contact_force,
            final_receiver_distance=distances[-1],
            final_object_pos=object_positions[-1],
            stages=stages,
            receiver_distances=distances,
            object_positions=object_positions,
        )
        rows.append(row)
        print("CHECKPOINT_ROLLOUT", json.dumps({k: v for k, v in row.items() if k not in ("stages", "receiver_distances", "object_positions")}), flush=True)

with open(os.path.join(args.output, "checkpoint_rollouts.json"), "w") as handle:
    json.dump(rows, handle, indent=2)
print("CHECKPOINT_RENDER_DONE", flush=True)
env.close()
simulation_app.close()
