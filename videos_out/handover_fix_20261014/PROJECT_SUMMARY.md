# Bimanual handover project summary — 6 November 2026

## What was built and corrected

The task is a cooperative two-arm, single-object handover in Isaac Lab. Either arm can be the giver at episode reset. Both slots use GEN3 arms; the `UR5e` identifier is a legacy slot name. The current agent uses a shared PPO policy with role-dependent observations/actions and staged coordination. It is not a pair of independently learning adversarial self-play agents.

- Corrected the initial joint poses and the giver-relative object placement. The original palms' contact sides faced mostly away from each other; the validated pose faces the palm fronts toward one another at about 0.32 m separation. Zero-action rollouts confirmed stable grip in both directions.
- Fixed both shared and legacy policy action heads being replaced after the Stable-Baselines3 optimizer was built. The optimizer now includes and updates those heads.
- Added the required GEN3 USD asset to the local checkout.
- Changed the bar from the original 0.25 g to a validated 5 g. A 50 g trial disturbed the starting grip and was rejected.
- Added object-escape/non-finite termination and corrected Isaac Lab's `(terminated, time_outs)` return order. Previously successful transfers could be interpreted as time-limit truncations.
- Added balanced giver-role metrics, true success-per-completed-episode counters, and separate curriculum-level logging.
- Kept all cluster experiments in `/home/hapi039h/codex_handover_20261014`; the original `/home/hapi039h/isaaclab` checkout was not modified.

The task has four progress stages: present the held bar at the midpoint (stage 1), receiver touches it (stage 2), receiver sustains a grasp while the giver starts opening (stage 3), and receiver remains grasping after release (stage 4). Curriculum level 4 is the full midpoint-start task. Stage and curriculum level are different quantities. The training success counter is `transfer_complete & ~failure` over completed episodes; slot 0/1 identify which arm starts as giver.

## Training history

| Run | Slurm job | Outcome |
|---|---:|---|
| Initial long run | 4416099 | Stopped at 37.96M transitions after rare extreme object-distance spikes. Curriculum remained at level 0 and success was sparse. Preserved 35.84M checkpoint. |
| Corrected 5 g run | 4419331 | Reached near-100% training success for both roles around 140–275M transitions. Slot 1 later collapsed to zero and collisions rose; stopped around 326M. Preserved strong 273.92M and late 325.12M checkpoints. |
| Low-learning-rate resume | 4426895 | Resumed 273.92M at curriculum 4, PPO learning rate `1e-5`, target KL `0.02`; completed at 600.00M. Training success recovered, but held-out start robustness was worse than the earlier checkpoint. |
| Arm-start randomization (v4) | 4442177 | Resumed 273.92M for 40M more transitions at learning rate `1e-5`, target KL `0.02`, curriculum 4. Randomized each arm's shoulder ±0.03 rad and elbow ±0.025 rad at reset, with 25% nominal starts; repositioned the bar in the giver's hand. Completed at 313.92M. Final training success rates were slot 0: 1.000, slot 1: 0.992. |

The v4 pilot (job 4441715) checked the intended joint ranges, 64 environments over 200 steps, and found no first-step termination, non-finite state, or sampled object-escape truncation. These were setup checks, not handover successes.

## Held-out results

The same 16 controlled variations (both giver roles, nominal starts, shoulder/elbow shifts, and ±6 mm object offsets) gave 15/16 for the 273.92M checkpoint, 13/16 for 600M, and **16/16 for v4**. Initial measured positions matched across the three evaluations.

For a harder paired test, all three deterministic policies were evaluated on the same 128 seeded starts, 64 per giver role. Each model saw identical initial joint angles and object positions. The shoulder range was ±0.06 rad and elbow range ±0.05 rad, twice v4's training ranges.

| Checkpoint | Overall stage-4 completions | Slot 0 giver | Slot 1 giver | First-step endings | Non-finite states |
|---|---:|---:|---:|---:|---:|
| 273.92M | 54/128 (42.2%) | 31/64 | 23/64 | 22 | 0 |
| 600M | 95/128 (74.2%) | 52/64 | 43/64 | 0 | 0 |
| **313.92M v4** | **108/128 (84.4%)** | **55/64** | **53/64** | **0** | **0** |

V4 succeeded where 600M failed in 24 matched episodes; 600M succeeded where v4 failed in 11. The 20 v4 failures mostly stopped at stage 2, pointing to receiver-grasp robustness as the next arm-pose issue. The evaluator's stored criterion was reaching stage 4 and terminating before timeout. It did not separately record a simultaneous failure on the terminal step, so these absolute counts may be slightly less strict than the environment's training success counter. All checkpoints used the same evaluation criterion. Full case-level data and failure stages are in `arm_heldout_4445714/comparison.md`.

## Current best model and artifacts

- Best checkpoint for tested arm-start robustness: `arm_randomization_v4/model_313921536_arm_v4.zip` (also preserved on Capella at `results/preserved_checkpoints/model_313921536_arm_v4.zip`).
- Labeled, slowed 16-episode video: `arm_v4_position_variations_4445628/arm_v4_all_16_labeled_slow.mp4`. Raw episodes and `episodes.json` are in the same directory.
- Strong earlier checkpoint video: `strong_checkpoint_4426905/strong_checkpoint_273920000_eight_episodes.mp4`.
- Original role-switching render: `multi_episode_4420579/role_switching_eight_episodes.mp4`.
- Setup validation: `REVIEW.md`, `setup_validation.json`, and `mass5g/setup_validation.json`.
- Training diagnoses: `TRAINING_DIAGNOSIS.md` and `TRAINING_RUN.md`.
- v4 W&B run: https://wandb.ai/hari3032003/bim_hand_dani_julio/runs/7bxp0oyb (run name `2026-10-21_01-38-49`). Earlier runs: https://wandb.ai/hari3032003/bim_hand_dani_julio/runs/9j3dyew5 and https://wandb.ai/hari3032003/bim_hand_dani_julio/runs/skmq0kr7.

## Current limits and next work

V4 is the best available checkpoint for the tested starts, but 84.4% at twice its trained pose range is not solved generalization. Continue the arm-pose phase by gradually widening reset randomization while retaining nominal starts, and evaluate with both roles and unseen seeds after each run. Select by worst-role held-out completion and physical failure rate, not training reward. Then vary object size; only after that, introduce multiple shapes with valid initial grasps and explicit geometry/size observations where needed. Arbitrary objects have not been trained or tested.

The stage-4 transition checks sustained receiver grasp after the scripted giver-opening phase; a stricter independent physical check of lost giver contact and retained receiver-only grasp would strengthen the benchmark. Success under this staged, shared-policy cooperative setup does not establish unrestricted coordination or independent-policy self-play. The heartbeat monitor was paused after the arm-start comparison was delivered.
