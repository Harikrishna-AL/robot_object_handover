from __future__ import annotations

from omni.isaac.lab.utils import configclass

from .bimanual_direct_env_cfg import BimanualDirectCfg

'''
Baseline task + inverse curriculum, as a SEPARATE task so BimanualDirectCfg /
BimanualDirect stay byte-identical and the with/without comparison is clean.

Purpose (2026-09-13): find out whether the curriculum helps at all, on a task
that already learns. The self-play env could not answer this -- diagnostics
there showed the receiving hand starts ~0.12m from the meeting point, of which
~0.10m is just the length of the hand, so there was no approach distance left
to curriculum over and all three implementations were no-ops. The baseline
gives the receiving arm a real ~0.45m approach and a randomised object pose,
so there is genuine difficulty to anneal.

What is annealed: the SIZE of the holder's pose randomisation. The baseline
samples the holder (and therefore the object) within +/-0.15m per axis and
+/-0.3/0.6/0.3 rad. At level 0 that range collapses to a point, so every
episode presents the object in the same place; the range then widens back to
the full baseline range, which the last level reproduces exactly.

Naming honesty: this is an easy-to-hard curriculum over the initial-state
DISTRIBUTION, not literally Florensa-style reverse curriculum (which starts
from states adjacent to the goal and walks backwards). Doing the literal
version needs the receiving arm placed near the object, which requires solving
IK at reset -- attempted three times in the self-play env and never made to
work (single differential step barely moved the arm; iterating it diverged
because write_joint_state_to_sim does not refresh body_state_w). Scaling the
sampling bounds needs no IK, no teleporting and no sim stepping.
'''


@configclass
class BimanualDirectICLCfg(BimanualDirectCfg):

    use_icl: bool = True

    # Multiplier on ee_pose_incs per level. 0.0 = object always in the same
    # place (easiest). The last entry MUST be 1.0 so the final level is the
    # unmodified baseline and the run stays comparable to a no-ICL control.
    icl_range_scale: list = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Pooled promote/demote over a rolling window of completed episodes.
    # Demotion included deliberately: it lets a run that over-promotes recover
    # instead of flatlining, which is how every previous run failed.
    icl_window: int = 200
    icl_promote_rate: float = 0.60
    icl_demote_rate: float = 0.15
