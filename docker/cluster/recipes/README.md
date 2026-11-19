# Working training recipe for the baseline

How to train `Isaac-Bimanual-Direct-reach-v0` (the handover method of Frau-Alfaro et
al., arXiv:2506.16822: UR5e gives, Kinova GEN3 receives) so that it actually learns. It
was developed on TU Dresden's Capella cluster and runs on any SLURM cluster with an
NVIDIA GPU and Apptainer or Singularity. See [Quick start](#quick-start-any-slurm-cluster).

## The one thing to know: don't pass `--max_iterations`

`sb3_ppo_cfg.yaml` asks for **600 million** timesteps:

```yaml
n_timesteps: !!float 6e8
```

but `train.py` silently overrides that whenever `--max_iterations` is given:

```python
if args_cli.max_iterations is not None:
    agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs
```

So a command like

```bash
--task Isaac-Bimanual-Direct-reach-v0 --num_envs 256 --max_iterations 1250
```

trains for `1250 × 16 × 256` = **5.12M steps — 0.85% of the intended budget.**

That matters because **the baseline does not leave the floor until roughly 10M
steps.** Below that it looks completely broken: reward flat around −12, zero
successes, `std` shrinking normally and `explained_variance` healthy. Nothing is
wrong; it simply has not trained yet.

If you are debugging a run that "won't learn", check the step count before
changing anything else.

## Measured reference curve

Job 4211348, 1024 envs, no video, single GPU:

| steps | `ep_rew_mean` | `success_rate` (per step) |
|------:|--------------:|--------------------------:|
|    5M |            −8 |                         0 |
|   11M |           +10 |                    1.4e-4 |
|   50M |          +146 |                    5.4e-4 |
|  150M |         +1050 |                    2.9e-3 |
|  300M |         +1190 |                    3.2e-3 |

Still climbing when the 24h wall clock stopped it, at half the intended budget.

### `success_rate` is NOT a success rate — read this before quoting it

The logged `metrics/success_rate` is `obj_reached_target.mean()` over the
**current step**: the fraction of environments sitting in the success state at
that instant. Reaching that state ends the episode, so an environment occupies
it for about one step out of ~200 — which divides the number you want by roughly
the episode length.

Measured on this same run's final checkpoint, over 13 rendered episodes:
**3 completed the handover, i.e. ~23%**, against a logged 3.2e-3 (0.3%). Off by
about 80x.

So: `success_rate` is fine as a *relative* progress signal between checkpoints of
the same run, and useless as an absolute number. To state a real success rate,
count completed episodes — `render_random_policy.py` prints a per-episode outcome
histogram for exactly this purpose. Do not put the logged figure in a report.

The curve is **not monotonic** — it has multi-checkpoint dips (e.g. 205 → 140 →
279, and 30.8 → 26.9 → 59.5) that look like plateaus but are troughs. Judge
progress over tens of millions of steps, not over a few log lines.

## Throughput

| configuration | steps/s | time for 6e8 |
|---|---:|---:|
| 256 envs, `--video --enable_cameras` | ~240 | ~29 days |
| 1024 envs, no video | **~3500** | **~48 h** |

Rendering roughly halves throughput, so train without `--video` and render
afterwards from a saved checkpoint.

`MaxTime` on the `capella` partition is `UNLIMITED` (default 8h), so a single
48-hour job can cover the full budget.

## Quick start (any SLURM cluster)

You need a SLURM cluster with one NVIDIA GPU per job (tested on an H100), Apptainer or
Singularity, about 10 GB in `$HOME` for the image, and about 20 GB of node-local scratch.
You don't need to install Isaac Sim. Untested: compute nodes without internet access (Isaac
Sim may try to reach NVIDIA's asset server at start-up).

**1. Get the code**

```bash
git clone -b baseline-working https://github.com/Harikrishna-AL/robot_object_handover.git
cd robot_object_handover
```

**2. Get the container image** (Isaac Sim 4.2 + Isaac Lab + SB3, about 8 GB). Pick one:

- *Copy a built one* (easiest): get `isaac-lab-base.sif` from someone who has it and put
  it at `$HOME/isaac-lab-base.sif`.
- *Build it*: this needs an NGC account (for `nvcr.io/nvidia/isaac-sim:4.2.0`) and
  `--fakeroot` support. Run it on a compute node, from the repo root:

  ```bash
  export APPTAINER_DOCKER_USERNAME='$oauthtoken' APPTAINER_DOCKER_PASSWORD=<your NGC API key>
  apptainer build --fakeroot $HOME/isaac-lab-base.sif docker/cluster/isaac-lab-base.def
  ```

**3. Smoke test** (about 10 minutes; checks that the GPU, the container and the task load):

```bash
SMOKE_ITERS=20 sbatch -p <partition> -A <account> --time=00:30:00 docker/cluster/recipes/train_baseline.sh
```

Look for `total_timesteps` rows in `baseline_<jobid>.log`, then `TRAIN_EXIT=0` at the end.

**4. Real run** (6e8 steps, about 48 h on one H100). Leave `SMOKE_ITERS` unset:

```bash
sbatch -p <partition> -A <account> docker/cluster/recipes/train_baseline.sh
# optional knobs: SEED=2 ENVS=1024 OUT_DIR=$HOME/runs SIF_PATH=... sbatch ...
```

If your partition caps jobs below 48 h, pass `--time` and expect a partial run. The script
does not resume.

Checkpoints and logs persist to `$HOME/runs/baseline_<jobid>/` every 15 minutes, so a job
killed at its time limit loses at most that. wandb logs offline unless `WANDB_API_KEY` is
set (in the environment or in `$HOME/.env.wandb`). Offline logs are in the output folder
for `wandb sync`. Track progress with `grep -a 'ep_rew_mean\|total_timesteps'
baseline_<jobid>.log` and compare against the reference curve above.

If it fails:
- `Segmentation fault` right after Kit starts: do not set `EXPERIENCE` to the full
  `omni.isaac.sim.python.kit` profile. The default headless profile works on newer drivers.
- Job hangs at start-up while binding paths: keep `$TMPDIR` and `$HOME` off slow or
  parallel network filesystems, or point `TMPDIR` at local disk.
- Some nodes fail with GPU or driver errors: add them to `--exclude=` on the sbatch line.

## Gotchas hit while getting this working

- **`$TMPDIR` is shared between jobs on the same node.** Any file staged there
  needs a job-scoped name or a concurrent job's copy collides and the job dies.
- **Sorting checkpoints by mtime does not work.** The periodic rescue re-copies
  every file, so all mtimes become identical and "newest N" returns an
  arbitrary N. Sort by the step number embedded in the filename.
- **Workspaces expire.** A `ws_allocate` workspace that lapses is deleted along
  with everything in it. `ws_extend` early, or keep checkpoints under `$HOME`.
- **Compute nodes could not see `/data/cat` or `/data/horse` on 2026-09-11**
  while `/home` was visible on all of them. If a job cannot find its files,
  check this before assuming a path bug.
- **Capella login nodes fail independently.** `login1` returning
  *"This account is currently not available"* while `login2` works is a node
  problem, not an account problem — try the other one first.
