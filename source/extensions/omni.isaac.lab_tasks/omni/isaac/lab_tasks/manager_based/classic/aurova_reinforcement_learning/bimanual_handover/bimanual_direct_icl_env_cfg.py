from __future__ import annotations

from omni.isaac.lab.utils import configclass

from .bimanual_direct_env_cfg import BimanualDirectCfg

'''
Baseline task + inverse curriculum, as a SEPARATE task so BimanualDirectCfg /
BimanualDirect stay byte-identical and the with/without comparison is clean.

WHAT IS ANNEALED: the receiving arm's STARTING DISTANCE to the object.
Level 0 starts it essentially on the object; the final level starts it at its
normal home pose, i.e. the unmodified baseline.

WHAT IS NOT ANNEALED: the object's pose randomisation. It stays at the full
baseline +/-0.15m / +/-0.3,0.6,0.3 rad at EVERY level.

That distinction is the whole point, and the previous version got it backwards.
It annealed the randomisation RANGE instead -- level 0 presented the object in
exactly the same place every episode. That is not a reverse curriculum (it
varies diversity, not distance to goal), and it failed catastrophically:
job 4243150 ran the curriculum correctly through all five levels and still
finished 540M steps at -21.5 reward with ZERO successes, against the plain
baseline's +1190 and 0.3%. The mechanism was visible in `std`: it collapsed
1.00 -> 0.121 within the first 25M steps while training on the zero-variability
level, and never recovered (0.06 at 540M). Collapsing the initial-state
distribution to a point removes everything there is to be uncertain about, the
policy goes deterministic, and it cannot adapt when variety returns.

Keeping the randomisation at full strength while shortening the reach is the
formulation that should not have that failure mode.
'''


@configclass
class BimanualDirectICLCfg(BimanualDirectCfg):

    use_icl: bool = True

    # Receiver arm joint configuration that places its hand ON the object, for
    # the object's NOMINAL pose. Solved offline (diag/solve_reach_pose.py, jobs
    # 4267693/4267798) by iterating the differential IK controller with real sim
    # steps between iterations, keeping the best iterate, and wrapping into
    # [-pi, pi].
    #
    # Verified, not assumed: this configuration was written back to the sim and
    # re-measured at 0.0548 m from the object (vs 0.4203 m at the home pose),
    # and checked to lie inside the joint limits. The other candidate solve
    # looked better raw (0.0306 m) but degraded to 0.6166 m when wrapped --
    # its out-of-limit values (7.5, -13.0 rad) were being clamped by PhysX on
    # write, so its reported distance never corresponded to those numbers.
    icl_near_object_arm_q: list = [
        -0.434602, 0.892648, -0.445363, -0.752578, -0.047977, 0.975372, 1.504364
    ]

    # Fraction of the way from the home pose to the on-object configuration, per
    # level. 1.0 = start on the object (easiest). The last entry MUST be 0.0 so
    # the final level is the unmodified baseline and stays comparable to a
    # no-ICL control.
    icl_reach_frac: list = [1.0, 0.75, 0.5, 0.25, 0.0]

    # Promotion on a rolling window of completed episodes, plus a step-based
    # safety net so the curriculum traverses its levels even if the success
    # threshold is never reached.
    icl_window: int = 200
    icl_promote_rate: float = 0.05
    icl_force_promote_steps: int = 25_000_000

    # DEMOTION DISABLED (0.0 means `rate < 0.0` is never true). On a task whose
    # success rate is near zero, a success-gated demotion fires unconditionally
    # and reverses every promotion ~200 episodes later -- verified by simulating
    # the update rule, which oscillated 0 -> 1 -> 0 indefinitely. That is why
    # jobs 4211349 and 4228865 never left level 0.
    icl_demote_rate: float = 0.0
