"""Which bodies does each robot's ground-contact sensor watch, and which ones touch the ground?"""
import argparse, json
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--task", default="Isaac-Bimanual-Bidirectional-xArm-v0")
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import torch, gymnasium as gym
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
cfg = parse_env_cfg(task_name=a.task, device=a.device, num_envs=2); cfg.use_reverse_curriculum = False
env = gym.make(a.task, cfg=cfg); u = env.unwrapped; env.reset()
env.step(torch.zeros((2, env.action_space.shape[-1]), device=u.device))
for k in u.cfg.keys:
    s = u.scene.sensors[f"{k.lower()}_w_ground"]
    f = s.data.force_matrix_w[0].abs().sum(dim=(-1, -2))           # per watched body, env 0
    print(f"GROUND_SENSOR {k}: {len(s.body_names)} bodies", flush=True)
    print("  touching ground:", json.dumps({n: round(float(v), 3) for n, v in zip(s.body_names, f) if v > 0}), flush=True)
    print("  watched:", s.body_names, flush=True)
    art = u.scene.articulations[k]
    z = art.data.body_state_w[0, :, 2] - u.scene.env_origins[0, 2]
    low = sorted(((float(zz), n) for n, zz in zip(art.body_names, z)))[:4]
    print("  lowest bodies (z, name):", [(round(zz, 3), n) for zz, n in low], flush=True)
env.close(); app.close()
