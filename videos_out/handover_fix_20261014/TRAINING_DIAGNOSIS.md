# Training diagnosis, 15 October 2026

Original long run: Slurm 4416099, cancelled after 37,961,728 transitions and preserved checkpoint at 35,840,000 transitions. Full stdout log is `train_4416099.log`. Checkpoint videos and measurements are in `checkpoint_35840000/`.

The run was not completely devoid of transfers: logged per-role stage 4 fractions were occasionally nonzero, and the best pooled 500-episode success window reached about 0.4%. But both roles spent most rollouts at stage 2 (receiver touch), sustained grasps were rare, and curriculum level never exceeded 0. A saved-policy deterministic rollout in each giving direction reached stage 2 but did not sustain a grip. The receiver moved away after touching in the slot-0 clip.

The training-batch hand-to-object distance was normally about 0.15-0.25 m, then exceeded 700 m near 37.9 million transitions. The single-environment checkpoint videos did not reproduce that rare extreme. The original object is only 0.00025 kg; the environment checked for object falling below a height threshold, but had no horizontal escape condition, and stage 2 remains latched after one touch. Thus a launched object could remain in a long stage-2 episode with a misleading stage fraction.

Another confirmed issue: BimanualBidirectional._get_dones returned (failure, timeout-or-success), whereas DirectRLEnv expects (terminated, time_out). Stable-Baselines3 consequently saw successful transfers as time-limit truncations and could bootstrap them incorrectly. The candidate changes correct this order and terminate if the bar is >0.6 m from both end effectors or the distance is non-finite.

A 0.05 kg object trial in Slurm job 4419322 was rejected. It stayed within 11 mm of the giver but generated 33-50 N of giver contact, shifted arm joints by 0.25-0.58 rad, and lowered the bar by 14-184 mm during a 4.5-second no-action hold. This is not a suitable starting grip under the existing contact/controller settings.

A 0.005 kg follow-up trial was submitted as job 4419328. The candidate remains isolated from the original /home/hapi039h/isaaclab checkout.

The 0.005 kg follow-up in Slurm job 4419328 passed: both roles held the bar for 300 zero-action steps with no resets, 5.0-5.6 mm maximum object motion relative to the giver, and -5.0/+1.6 mm final height change. Palm normals stayed within 9.1° of facing the other palm. This bar is still light, but it is twenty times heavier than the original and did not disturb the validated grip like the 50 g version. The corrected replacement run was submitted and released as job 4419331.
