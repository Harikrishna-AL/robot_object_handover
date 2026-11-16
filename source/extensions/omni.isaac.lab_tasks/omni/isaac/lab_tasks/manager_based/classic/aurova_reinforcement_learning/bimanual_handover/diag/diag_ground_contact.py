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
env = gym.make(a.task, cfg=cfg); u = env.unwrapped
names = list(u.cfg.contact_sensors_dict.keys())
print("CONTACT_COLUMNS first 4:", names[:4], "... last:", names[-1], "| total", len(names), flush=True)
def report(tag):
    gs = {k: float(u.scene.sensors[f"{k.lower()}_w_ground"].data.force_matrix_w.abs().sum()) for k in u.cfg.keys}
    hot = [names[i] for i in range(len(names)) if bool(u.contacts[:, i].any())]
    print(f"STEP {tag}: contacts[:,0:2]={u.contacts[:, :2].tolist()} ground_force_sum={gs} "
          f"episode_len={u.episode_length_buf.tolist()} nonzero_columns={hot[:6]}", flush=True)
obs, _ = env.reset(); report("after_reset")
z = torch.zeros((2, env.action_space.shape[-1]), device=u.device)
for t in range(4):
    _, _, term, tout, info = env.step(z)
    print(f"   -> step {t}: terminated={term.tolist()} time_out={tout.tolist()} trunc_ground={info.get('log', {}).get('metrics/trunc_ground_contact')}", flush=True)
    report(t)
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
