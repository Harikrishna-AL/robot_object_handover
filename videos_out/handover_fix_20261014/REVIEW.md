# Bimanual handover environment review — 14 October 2026

The corrected zero-action environment previews are `left_slot_gives_setup.mp4` and `right_slot_gives_setup.mp4`. No RL training was started. Cluster checks ran in `/home/hapi039h/codex_handover_20261014`; the original `/home/hapi039h/isaaclab` tree was not changed.

## What was wrong

- Both slots use GEN3 arms; `UR5e` is a legacy slot name. The old reset used one differential-IK step toward old Cartesian targets. At reset, the real palm fronts faced mostly away from one another (about 170° and 156° from the inward direction).
- Earlier diagnostics measured palm-to-fingertip extension and called it palm facing. That proxy can point inward while the contact side of the palm points sideways or outward. The corrected diagnostics use the palm link's +X normal, established from the positive-X palm contact plates in the URDF.
- The shared and legacy actor-critic policies replaced their action heads *after* Stable-Baselines3 built the optimizer. The replacement head's parameters were absent from the optimizer, so the final action layer did not train. Both policies now rebuild their optimizer after installing the head.
- The local checkout was missing `config/usd/gen3_4f_arm2.usd`, required by the environment configuration. A copy of the asset used on the cluster is now in that path.

## Corrected setup and checks

The environment now resets both arms to a validated joint configuration, bypassing the one-step IK reset, and places the bar at a re-solved end-effector-relative offset. Both true palm normals initially face the other palm (0° measured angle), with 0.32 m palm separation. The matching joint pose and object placement were selected in a 32-case simulation search: four postures × four offsets × both giving roles. The selected pair held the unwelded bar for 900 physics steps (4.5 simulated seconds) with under 5.0 mm motion relative to the giver, zero arm collision force, and no receiver contact. Raw search data: `true_palm_search.json`; selected row: `selected_pose.json`.

The final configuration also passed the actual environment reset/action path, 300 zero-action policy steps (4.5 simulated seconds) in each giving direction, with no automatic episode reset:

| Check | Left slot gives | Right slot gives |
|---|---:|---:|
| Maximum bar drift relative to giver | 4.45 mm | 6.87 mm |
| Final bar height change | -4.30 mm | +1.09 mm |
| Maximum palm-facing angle during rollout, giver / receiver | 0.84° / 0.49° | 0.67° / 2.29° |
| Initial receiver fingertips to bar | 0.333 m | 0.333 m |
| Receiver contact force | 0 N | 0 N |
| Maximum repeated-reset joint error (five resets) | 0 rad | 0 rad |

These are setup and grip stability checks, **not learned handover success rates**. Both rollouts remained in stage 0 because the policy took zero actions. Full measurements are in `setup_validation.json`. The two videos show the exact checked rollouts.

The cluster also passed `diag/check_policy_optimizer.py` for both policies: all current parameters belong to the optimizer, a synthetic optimizer step changes the action-head weights, and the shared policy passes its arm-swap action-mean check. This is a policy wiring check, not PPO training.

## Recommended next work

1. Train a short balanced baseline with this geometry and evaluate each giving direction separately. Track first receiver contact, actual giver release, post-release bar retention, and drop/collision rates. Old checkpoints are not directly comparable after the geometry change.
2. Require physical loss of giver contact and sustained receiver-only retention for success. A commanded open hand alone does not establish transfer.
3. Save curriculum level and counters with checkpoints; promote only after both directions meet the threshold, with a dwell period to prevent oscillation.
4. Check object physics: the 35 cm bar currently has mass 0.00025 kg (0.25 g). Validate a realistic mass and friction in a separate experiment after the corrected-geometry baseline.
5. The present implementation is shared-policy cooperative learning with randomized roles, a central value estimate, gated movement stages, and scripted giver closure/release. After a baseline, relax the gates gradually if unrestricted coordination is the goal; evaluate cross-play with saved partners if independent policies are desired.
