# Working training recipe — Capella

How to train `Isaac-Bimanual-Direct-reach-v0` so that it actually learns.

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

| steps | `ep_rew_mean` | `success_rate` |
|------:|--------------:|---------------:|
|    5M |            −8 |              0 |
|   11M |           +10 |         1.4e-4 |
|   50M |          +146 |         5.4e-4 |
|  150M |         +1050 |         2.9e-3 |
|  300M |         +1190 |         3.2e-3 |

Still climbing when the 24h wall clock stopped it, at half the intended budget.

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

## Usage

```bash
# build the container once (see ../isaac-lab-base.def), then:
export ISAACLAB_TREE=$HOME/isaaclab          # defaults to $HOME/isaaclab
export SIF_PATH=$HOME/isaac-lab-base.sif     # defaults to $HOME/isaac-lab-base.sif
sbatch docker/cluster/recipes/train_baseline.sh
```

Checkpoints land in `$HOME/runs/<jobid>/` and are rescued every 15 minutes, so
a job killed at its time limit loses at most that.

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
