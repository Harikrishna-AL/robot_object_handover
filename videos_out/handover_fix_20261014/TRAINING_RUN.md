# Bimanual handover training runs

The first long run, Slurm 4416099, was cancelled at 37,961,728 transitions after extreme object-distance spikes. Its latest preserved checkpoint is 35,840,000 steps. The complete stdout is `train_4416099.log`, and both deterministic and stochastic checkpoint videos are in `checkpoint_35840000/`.

The corrected run is Slurm **4419331**, submitted and released on 15 October 2026 (India time). Its state at release was PENDING in Slurm. It uses the isolated candidate source at `/home/hapi039h/codex_handover_20261014/candidate/`; the original `/home/hapi039h/isaaclab` checkout remains unchanged.

- Task: Isaac-Bimanual-Bidirectional-v0
- 512 environments, PPO budget 600,000,000 transitions, 7-day Slurm wall limit
- 5 g bar, validated in both giving directions for 300 zero-action steps (4.5 simulated seconds)
- Episode termination on object escape (>0.6 m from both end effectors) or non-finite distance
- Correct `DirectRLEnv` return order: terminal success/failure versus time limit
- Persistent checkpoints every 5,000 environment steps (about 2.56 million transitions)
- W&B offline; persistent TensorBoard logs and checkpoints in `/home/hapi039h/codex_handover_20261014/training_outputs_v2/logs/sb3/Isaac-Bimanual-Bidirectional-v0/`
- Slurm stdout: `/home/hapi039h/codex_handover_20261014/train_v2_4419331.log`
- Launcher: `run_long_training_v2.sh`

The 5 g setup videos and measurements are in `mass5g/`. The automatic thread monitor checks the replacement run every 30 minutes and reports meaningful changes.
