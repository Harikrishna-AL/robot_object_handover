import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Bimanual handover — Kaggle smoke test

Checks, in order, whether this project can run on Kaggle at all. **Each step prints a PASS/FAIL line — stop at the first FAIL** and send it back; later steps depend on earlier ones.

**Before running — Kaggle settings (right-hand panel):**
1. *Accelerator* → **GPU T4 x2** (not P100 — too old for Isaac Sim)
2. *Internet* → **On** (needs a phone-verified Kaggle account)

Then *Run All*. Total time is roughly 25–40 min, mostly downloading Isaac Sim (~15 GB).

| step | what it answers |
|---|---|
| 1 | GPU, driver, RAM, disk are usable |
| 2 | **Vulkan driver present** — the most likely blocker; Isaac Sim won't start without it |
| 3–5 | Python 3.10 env, Isaac Sim 4.2, this repo installed |
| 6 | Isaac Sim actually starts and the handover env steps |
| 7 | The v4 checkpoint loads and does handovers, per giving arm |
| 8 | Steps/second at training scale — tells us if training here is realistic |
""")

code(r"""
# STEP 1 -- hardware
!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
!grep PRETTY_NAME /etc/os-release
!python --version
!free -g | head -2
!df -h / /kaggle/working | tail -2
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
print("\nSTEP 1:", "PASS" if ("T4" in gpu or "L4" in gpu or "A100" in gpu) else f"FAIL -- unsupported/absent GPU: {gpu!r} (P100 will not work)")
""")

code(r"""
# STEP 2 -- Vulkan. Isaac Sim's Kit runtime loads Vulkan even headless.
!apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y libvulkan1 vulkan-tools > /dev/null 2>&1; echo apt done
import glob, os, json
icds = glob.glob("/usr/share/vulkan/icd.d/nvidia*.json") + glob.glob("/etc/vulkan/icd.d/nvidia*.json")
print("NVIDIA Vulkan ICD files:", icds)
if not icds:
    # The driver library may be mounted even if its ICD manifest is not; if so, write one.
    libs = [p for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/nvidia/lib64", "/usr/lib"]
            for p in glob.glob(f"{d}/libGLX_nvidia.so.0")]
    print("libGLX_nvidia.so.0 found at:", libs)
    if libs:
        os.makedirs("/usr/share/vulkan/icd.d", exist_ok=True)
        with open("/usr/share/vulkan/icd.d/nvidia_icd.json", "w") as f:
            json.dump({"file_format_version": "1.0.0",
                       "ICD": {"library_path": libs[0], "api_version": "1.3.0"}}, f)
        print("wrote /usr/share/vulkan/icd.d/nvidia_icd.json")
!vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|apiVersion|ERROR" | head -8
out = subprocess.run("vulkaninfo --summary", shell=True, capture_output=True, text=True).stdout
print("\nSTEP 2:", "PASS -- NVIDIA GPU visible to Vulkan" if "NVIDIA" in out else
      "FAIL -- no NVIDIA Vulkan device. Isaac Sim will almost certainly not start on this machine.")
""")

code(r"""
# STEP 3 -- Python 3.10 env (Isaac Sim 4.2 pip wheels require exactly 3.10)
!pip -q install uv
!uv venv /root/isaacenv --python 3.10 --seed -q
!/root/isaacenv/bin/python --version
import os
os.environ["PATH"] = "/root/isaacenv/bin:" + os.environ["PATH"]
os.environ["VIRTUAL_ENV"] = "/root/isaacenv"
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["WANDB_MODE"] = "offline"
os.environ["ISAACLAB_PATH"] = "/workspace/isaaclab"
PY = "/root/isaacenv/bin/python"
print("STEP 3: PASS" if os.path.exists(PY) else "STEP 3: FAIL")
""")

code(r"""
# STEP 4 -- torch + Isaac Sim 4.2 (large download; ~10-20 min)
!uv pip install --python /root/isaacenv/bin/python -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
!uv pip install --python /root/isaacenv/bin/python -q "isaacsim==4.2.0.2" "isaacsim-extscache-physics==4.2.0.2" "isaacsim-extscache-kit==4.2.0.2" "isaacsim-extscache-kit-sdk==4.2.0.2" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
# isaaclab.sh looks for isaacsim-rl specifically; install it if the meta-package did not pull it in
!/root/isaacenv/bin/python -m pip list 2>/dev/null | grep -iE "^isaacsim-rl |^torch " || uv pip install --python /root/isaacenv/bin/python -q "isaacsim-rl==4.2.0.2" "isaacsim-replicator==4.2.0.2" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
r = subprocess.run([PY, "-c", "import torch, isaacsim; print(torch.__version__, torch.cuda.is_available())"], capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
print("STEP 4:", "PASS" if "True" in r.stdout else "FAIL")
""")

code(r"""
# STEP 5 -- this repo. robots_cfg.py hardcodes /workspace/isaaclab/... USD paths (the cluster container
# layout), so the clone is symlinked there rather than editing the code.
!rm -rf /root/handover && git clone -q --depth 1 -b tmp-cluster-fixes https://github.com/Harikrishna-AL/robot_object_handover.git /root/handover
!mkdir -p /workspace && ln -sfn /root/handover /workspace/isaaclab
# Install the extensions directly. isaaclab.sh runs with `set -e` and calls the cluster container's
# own python (/workspace/isaaclab/_isaac_sim/kit/python), which does not exist here.
!cd /workspace/isaaclab && uv pip install --python /root/isaacenv/bin/python -e source/extensions/omni.isaac.lab -e source/extensions/omni.isaac.lab_assets -e "source/extensions/omni.isaac.lab_tasks[sb3]" > /root/isaaclab_install.log 2>&1; echo exit=$?; tail -3 /root/isaaclab_install.log
!uv pip install --python /root/isaacenv/bin/python -q wandb
r = subprocess.run([PY, "-c", "import omni.isaac.lab_tasks, stable_baselines3, wandb; print('ok')"], capture_output=True, text=True)
print(r.stdout, r.stderr[-1500:])
# The repo has no Isaac Lab .kit app files, so point at Isaac Sim's own kit -- the same one the cluster used.
KIT = subprocess.run([PY, "-c", "import isaacsim, os, glob; d=os.path.dirname(isaacsim.__file__); "
                      "print((glob.glob(d+'/**/apps/omni.isaac.sim.python.kit', recursive=True) or [''])[0])"],
                     capture_output=True, text=True).stdout.strip()
print("kit file:", KIT)
print("STEP 5:", "PASS" if ("ok" in r.stdout and KIT) else "FAIL")
""")

code(r"""
%%writefile /root/smoke_test.py
# Starts Isaac Sim, builds the handover env, and (optionally) runs the v4 checkpoint with per-role counts.
import argparse, os, sys, time
from omni.isaac.lab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--checkpoint", type=str, default="")
p.add_argument("--throughput", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch, gymnasium as gym
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks  # noqa
TASK = "Isaac-Bimanual-Bidirectional-v0"
cfg = parse_env_cfg(task_name=TASK, device=args.device, num_envs=args.num_envs)
env = gym.make(TASK, cfg=cfg); u = env.unwrapped
obs, _ = env.reset()
print(f"ENV_OK num_envs={u.num_envs} action_dim={env.action_space.shape[-1]}", flush=True)

if args.throughput:
    a = torch.zeros((u.num_envs, env.action_space.shape[-1]), device=u.device)
    for _ in range(20): env.step(a)                       # warm-up
    torch.cuda.synchronize(); t = time.time()
    for _ in range(args.steps): env.step(a)
    torch.cuda.synchronize(); dt = time.time() - t
    sps = args.steps * u.num_envs / dt
    print(f"THROUGHPUT num_envs={u.num_envs} env_steps_per_s={sps:.0f}  "
          f"(cluster H100 was ~3500)  -> 40M steps would take {40e6/sps/3600:.1f} h", flush=True)
    print(f"GPU_MEM_GB={torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)

elif args.checkpoint:
    sys.path.insert(0, "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/"
                       "manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train")
    import networks  # noqa -- registers the custom policy class for unpickling
    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint, device=u.device)
    # A checkpoint does not carry the curriculum level; pin the real task (level 4) explicitly.
    u.curriculum_level = 4
    u._update_curriculum = lambda success, done: None
    obs, _ = env.reset()
    role = u.giver_is_gen3.clone(); peak = u.stage.clone()
    res = {0: [0, 0], 1: [0, 0]}                           # slot -> [episodes, successes]
    for _ in range(args.steps):
        act, _ = model.predict(obs["policy"].cpu().numpy(), deterministic=True)
        obs, rew, term, trunc, info = env.step(torch.as_tensor(act, device=u.device))
        peak = torch.maximum(peak, u.stage)
        done = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
        for i in done.tolist():
            r = int(role[i]); res[r][0] += 1; res[r][1] += int(peak[i] >= 4)
        role[done] = u.giver_is_gen3[done]; peak[done] = u.stage[done]
    for r, name in ((0, "slot 0 gives"), (1, "slot 1 gives")):
        n, s = res[r]
        print(f"POLICY {name}: {s}/{n} episodes reached stage 4", flush=True)
else:
    a = torch.zeros((u.num_envs, env.action_space.shape[-1]), device=u.device)
    for _ in range(args.steps): env.step(a)
    print(f"STEP_OK ran {args.steps} zero-action steps; stage per env = {u.stage.tolist()}", flush=True)

env.close(); app.close()
""")

code(r"""
# STEP 6 -- does Isaac Sim start and the env step? (first launch is slow: shader/extension caches)
!cd /workspace/isaaclab && timeout 1500 /root/isaacenv/bin/python /root/smoke_test.py --num_envs 4 --steps 50 --experience "$KIT" > /root/step6.log 2>&1; echo exit=$?
!grep -aE "ENV_OK|STEP_OK|Error|error:|Traceback|vk|Vulkan" /root/step6.log | grep -v "Failed to create directory" | tail -20
log = open("/root/step6.log", errors="ignore").read()
print("\nSTEP 6:", "PASS" if "STEP_OK" in log else "FAIL -- see /root/step6.log (send me the last ~60 lines)")
""")

code(r"""
# STEP 7 -- v4 checkpoint at the real task (level 4), 32 envs
CKPT = "/workspace/isaaclab/videos_out/handover_fix_20261014/arm_randomization_v4/model_313921536_arm_v4.zip"
!cd /workspace/isaaclab && timeout 2400 /root/isaacenv/bin/python /root/smoke_test.py --num_envs 32 --steps 1500 --checkpoint $CKPT --experience "$KIT" > /root/step7.log 2>&1; echo exit=$?
!grep -aE "ENV_OK|POLICY|Traceback|Error" /root/step7.log | tail -12
print("\nSTEP 7:", "PASS" if "POLICY" in open("/root/step7.log", errors="ignore").read() else "FAIL -- see /root/step7.log")
""")

code(r"""
# STEP 8 -- training-scale throughput. Tries 512 then 1024 envs; T4 has 16 GB so 1024 may not fit.
for n in (512, 1024):
    !cd /workspace/isaaclab && timeout 1500 /root/isaacenv/bin/python /root/smoke_test.py --num_envs {n} --steps 200 --throughput --experience "$KIT" > /root/step8_{n}.log 2>&1; echo n={n} exit=$?
    !grep -aE "THROUGHPUT|GPU_MEM|out of memory|Traceback" /root/step8_{n}.log | tail -4
""")

md("""
## What to send back
The **STEP lines** (and for any FAIL, the last ~60 lines of that step's `/root/stepN.log`).
- Fails at **2** → Kaggle's machine has no Vulkan driver; Isaac Sim can't run here. Fall back to a lab machine or the cluster.
- Passes **6–7** → evaluation/video work is possible on Kaggle.
- **Step 8** steps/s tells us whether a training run fits Kaggle's 12 h sessions and 30 GPU-h/week.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("handover_kaggle_smoke_test.ipynb", "w"), indent=1)
print("written", len(cells), "cells")
