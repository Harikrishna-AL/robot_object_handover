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
    # Lowered 0.60 -> 0.05 (2026-09-16). At 0.60 the curriculum never advanced:
    # job 4211349 spent all 290M steps pinned at level 0 while the baseline
    # reached only 0.3% success, so the "curriculum" was really just a fixed
    # easiest setting -- and it scored -29 reward with ZERO successes against
    # the baseline's +1190. A threshold the task cannot reach is not a
    # curriculum.
    icl_promote_rate: float = 0.05
    # DEMOTION DISABLED (0.0 means `rate < 0.0` is never true).
    #
    # Demotion was my addition, and it silently broke both ICL runs. On a task
    # whose success rate sits near zero, `rate < demote_rate` is ALWAYS true,
    # so every promotion is reversed ~200 episodes later and the level
    # oscillates 0 -> 1 -> 0 forever. Verified by simulating the update rule:
    # with forced promotion at 25M steps and demotion at 1%, the level returns
    # to 0 within 200k steps of every promotion and never exceeds 1.
    #
    # That is why job 4211349 sat at level 0 for 290M steps and 4228865 for
    # 131M. A reverse curriculum is meant to climb monotonically; re-enable
    # this only once success rates are high enough for it to mean something.
    icl_demote_rate: float = 0.0

    # Safety net so the curriculum traverses its levels even if the success
    # threshold is never reached. Without it the run silently degenerates into
    # single-difficulty training, which is what 4211349 produced.
    #
    # Counted in ENVIRONMENT STEPS, not episodes. The episode-counted version
    # (job 4228865) never fired: at 131M steps with ~199-step episodes that is
    # ~660k completed episodes against a threshold of 20k, so it should have
    # maxed out the level ~30 times over, yet icl_level stayed 0. Rather than
    # keep guessing at why the episode counter did not accumulate, this uses a
    # counter incremented once per _get_dones call, which cannot be ambiguous.
    # 25M steps/level traverses all 5 levels by ~100M, leaving the bulk of the
    # 6e8 budget at full difficulty.
    icl_force_promote_steps: int = 25_000_000
