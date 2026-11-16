"""Physics check of a handover start pose, arms held still (zero actions), weld off.

Same acceptance tests Codex used for the GEN3 posture, per giving slot:
  - palm facing: angle between each palm's front (+X of hand_palm_link) and the
    direction to the other palm, and the palm separation
  - grip: object drift relative to the giver's palm over the run (Codex: < 5 mm)
  - no arm-to-arm contact, no early terminations
Run on BOTH tasks (GEN3 and xArm) so the xArm numbers have a reference.
"""
import argparse, json, math
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Bimanual-Bidirectional-xArm-v0")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--steps", type=int, default=900)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import torch, gymnasium as gym
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
cfg = parse_env_cfg(task_name=a.task, device=a.device, num_envs=a.num_envs)
cfg.use_reverse_curriculum = False
env = gym.make(a.task, cfg=cfg); u = env.unwrapped
obs, _ = env.reset()
keys = u.cfg.keys
palm = {k: u.scene.articulations[k].find_bodies("hand_palm_link")[0][0] for k in keys}
origins = u.scene.env_origins
def palms():
    out = {}
    for k in keys:
        st = u.scene.articulations[k].data.body_state_w[:, palm[k]]
        pos = st[:, :3] - origins; q = st[:, 3:7]
        w, x, y, z = q.unbind(-1)   # +X axis of the palm frame from the quaternion (w, x, y, z)
        xax = torch.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], -1)
        out[k] = (pos, xax)
    return out
P = palms(); k0, k1 = keys
d01 = P[k1][0] - P[k0][0]; sep = d01.norm(dim=-1)
ang0 = torch.rad2deg(torch.acos((P[k0][1] * d01 / sep[:, None]).sum(-1).clamp(-1, 1)))
ang1 = torch.rad2deg(torch.acos((P[k1][1] * -d01 / sep[:, None]).sum(-1).clamp(-1, 1)))
role = u.giver_is_gen3.clone()      # True: slot 1 gives
obj = u.scene.rigid_objects["object"]
def rel_obj():
    o = obj.data.root_pos_w - origins
    giver_pos = torch.where(role[:, None], palms()[k1][0], palms()[k0][0])
    return o - giver_pos
r0 = rel_obj(); z0 = (obj.data.root_pos_w - origins)[:, 2].clone()
zero = torch.zeros((u.num_envs, env.action_space.shape[-1]), device=u.device)
max_drift = torch.zeros(u.num_envs, device=u.device); arm_contact = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
ended = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
for t in range(a.steps):
    _, _, term, tout, _ = env.step(zero)
    ended |= (term | tout)
    live = ~ended
    max_drift = torch.where(live, torch.maximum(max_drift, (rel_obj() - r0).norm(dim=-1)), max_drift)
    arm_contact |= (u.contacts[:, -1] > 0) & live
z1 = (obj.data.root_pos_w - origins)[:, 2]
res = {"task": a.task, "num_envs": u.num_envs, "steps": a.steps,
       "palm_separation_m": [round(float(sep.min()), 4), round(float(sep.max()), 4)],
       "palm_facing_deg_slot0": round(float(ang0.max()), 2), "palm_facing_deg_slot1": round(float(ang1.max()), 2),
       "slot0_palm_local_xyz": [round(float(v), 3) for v in P[k0][0][0]], "slot1_palm_local_xyz": [round(float(v), 3) for v in P[k1][0][0]],
       "object_z0": round(float(z0.mean()), 4)}
for r, name in ((False, "slot0_gives"), (True, "slot1_gives")):
    m = role == r
    if m.any():
        res[name] = {"envs": int(m.sum()), "ended_early": int((ended & m).sum()),
                     "max_object_drift_mm": round(float(max_drift[m].max()) * 1000, 2),
                     "object_drop_mm": round(float((z0 - z1)[m & ~ended].max()) * 1000, 2) if (m & ~ended).any() else None,
                     "arm_contact_envs": int((arm_contact & m).sum())}
ok = all(res[n]["ended_early"] == 0 and res[n]["max_object_drift_mm"] < 5.0 and res[n]["arm_contact_envs"] == 0
         for n in ("slot0_gives", "slot1_gives") if n in res) and max(res["palm_facing_deg_slot0"], res["palm_facing_deg_slot1"]) < 2.0
res["VERDICT"] = "PASS" if ok else "FAIL"
print("VALIDATE_RESULT " + json.dumps(res), flush=True)
env.close(); app.close()
