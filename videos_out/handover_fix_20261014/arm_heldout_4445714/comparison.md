# Held-out arm-start comparison

All three policies were tested on the same 128 seeded starts (64 per giver role) in the same pose-randomized environment. Shoulder offsets were sampled within ±0.06 rad and elbow offsets within ±0.05 rad, twice the v4 training range. Each policy used deterministic actions. Episode seed, giver role, reset attempts, initial arm joint angles, and initial object position matched exactly across policies.

| Checkpoint | Overall | Slot 0 giver | Slot 1 giver | Worst-role success | First-step endings | Timeouts | Nonfinite |
|---|---:|---:|---:|---:|---:|---:|---:|
| 273.92M original | 54/128 (42.2%) | 31/64 (48.4%) | 23/64 (35.9%) | 23/64 (35.9%) | 22 | 18 | 0 |
| 600M continued | 95/128 (74.2%) | 52/64 (81.2%) | 43/64 (67.2%) | 43/64 (67.2%) | 0 | 18 | 0 |
| 313.92M pose-randomized | 108/128 (84.4%) | 55/64 (85.9%) | 53/64 (82.8%) | 53/64 (82.8%) | 0 | 12 | 0 |

The evaluator counts an episode as successful when it reached stage 4 and terminated before the time limit; this is the same criterion for all three checkpoints. The environment's strict success counter additionally excludes simultaneous failure on the final step, which this saved evaluation did not record.

- 273.92M original failed stages: stage 0: 26, stage 1: 23, stage 2: 24, stage 3: 1
- 600M continued failed stages: stage 0: 13, stage 1: 2, stage 2: 6, stage 3: 12
- 313.92M pose-randomized failed stages: stage 0: 4, stage 1: 2, stage 2: 13, stage 3: 1

Paired outcomes (original, 600M, v4; 1=success): {'000': 9, '001': 16, '010': 11, '011': 38, '101': 8, '111': 46}.
V4 succeeded where 600M failed in 24 episodes; 600M succeeded where v4 failed in 11. V4 succeeded in every case the original passed, plus 54 more.
The 16 controlled-start video test retained both nominal giver roles: v4 passed 16/16, original 15/16, and 600M 13/16.

Recommendation: keep v4 as the current best arm-start checkpoint. Continue arm-pose training with a gradual range increase and nominal-start retention before moving to object-size variation; 20/128 held-out episodes still failed at twice the trained pose range, mostly at stage 2.
