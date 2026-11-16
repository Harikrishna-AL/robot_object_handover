"""Does an end-effector command move the hand the same way on GEN3 and xArm?

Each arm gets a constant +1 action on ONE channel (x, y or z translation, or
nothing) for N steps with stage gating off so both arms may move. Reports the
achieved end-effector displacement expressed in the EE's START frame (the frame
the command is given in), per arm. A healthy IK gives a displacement mostly
along the commanded axis with similar magnitude on both robots; a frame or
indexing bug shows up as the wrong direction or scale.
"""
import argparse, json
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--task", default="Isaac-Bimanual-Bidirectional-xArm-v0")
p.add_argument("--steps", type=int, default=15)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import torch, gymnasium as gym
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils.math import quat_rotate_inverse
import omni.isaac.lab_tasks  # noqa
cfg = parse_env_cfg(task_name=a.task, device=a.device, num_envs=4)
cfg.use_reverse_curriculum = False; cfg.stage_gate_arms = False
env = gym.make(a.task, cfg=cfg); u = env.unwrapped
A = env.action_space.shape[-1]; per = A // 2
print(f"IK_SETUP task={a.task} action_dim={A} translation_scale={u.cfg.translation_scale}", flush=True)
def ee(i):
    k = u.cfg.keys[i]; return u.scene.articulations[k].data.body_state_w[:, u.ee_jacobi_idx[i] + 1, :7]
out = {}
for name, ch in (("none", None), ("x", 0), ("y", 1), ("z", 2)):
    env.reset(); zero = torch.zeros((u.num_envs, A), device=u.device)
    for _ in range(2): env.step(zero)          # settle past the first-step reset artefacts
    start = [ee(i).clone() for i in range(2)]
    act = zero.clone()
    if ch is not None:
        act[:, ch] = 1.0; act[:, per + ch] = 1.0
    for _ in range(a.steps): env.step(act)
    res = {}
    for i in range(2):
        d_w = ee(i)[:, :3] - start[i][:, :3]
        d_ee = quat_rotate_inverse(start[i][:, 3:7], d_w)     # displacement in the EE's start frame
        rot = torch.rad2deg(2 * torch.acos((ee(i)[:, 3:7] * start[i][:, 3:7]).sum(-1).abs().clamp(max=1)))
        res[f"arm{i}"] = {"d_ee_mm": [round(float(v) * 1000, 1) for v in d_ee.mean(0)],
                          "rot_deg": round(float(rot.mean()), 2)}
    out[name] = res
    print(f"IK_CMD {name}: " + json.dumps(res), flush=True)
print("IK_RESULT " + json.dumps(out), flush=True)
env.close(); app.close()
