from __future__ import annotations

import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat

from .bimanual_direct_env_cfg import BimanualDirectCfg
from .robots_cfg import GEN3_4f_CFG

'''
                ############## IMPORTANT #################
Round-trip handover on top of BimanualDirectCfg's UR5e/GEN3 pair: phase 0 is
BimanualDirectCfg's task unchanged (GEN3 grasps the object UR5e is holding),
phase 1 mirrors it with roles swapped (UR5e grasps the object back from GEN3).
Both arms are policy-driven every simulation step; SharedArmActorCriticPolicy
(train/networks.py) applies one set of weights to each arm's self-centric
observation block, so what GEN3 learns approaching/grasping in phase 0 is the
same skill UR5e needs in phase 1.
'''


# Per-arm links whose contact with the object matters for grasp-quality reward.
# Same set BimanualDirectCfg.update_collisions() wires for GEN3 only; mirrored
# here for both arms. Order matters -- reward code indexes into this block
# positionally (see BimanualBidirectional._phase_reward).
FINGER_LINKS = [
    "finger_1_contact_1_link", "finger_1_contact_2_link", "finger_1_contact_3_tip_link",
    "finger_2_contact_5_link", "finger_2_contact_6_link", "finger_2_contact_7_tip_link",
    "finger_3_contact_9_link", "finger_3_contact_10_link", "finger_3_contact_11_tip_link",
    "finger_4_contact_14_link", "finger_4_contact_15_tip_link",
    "hand_link_14__link", "hand_link_15__link", "hand_link_15__link_tip_link",
]

# Grasp-quality weight per FINGER_LINKS entry -- identical to the weights
# BimanualDirectCfg.update_collisions() uses for GEN3's equivalent 14 links,
# reused as-is for UR5e since both arms carry the same hand hardware.
FINGER_WEIGHTS = torch.tensor([
    0.65, 0.65, 0.4,
    0.65, 0.65, 0.4,
    0.65, 0.65, 0.4,
    0.65, 0.65,
    0.65, 0.65, 0.65,
])
assert FINGER_WEIGHTS.shape[0] == len(FINGER_LINKS)

PALM_WEIGHT = 0.15
HAND_WEIGHT = 0.0  # broad/redundant sensor, unweighted in the original task too

# Per-arm local contact block width: fingers + palm + hand.
ARM_CONTACT_WIDTH = len(FINGER_LINKS) + 2


_GEN3_LINK_NAMES = ['base_link', 'shoulder_link', 'half_arm_1_link', 'half_arm_2_link', 'forearm_link', 'spherical_wrist_1_link', 'spherical_wrist_2_link', 'bracelet_link', 'end_effector_link', 'hand_palm_link', 'hand_link_0_0_link', 'hand_link_12__link', 'hand_link_4_0_link', 'hand_link_8_0_link', 'hand_link_1_0_link', 'hand_link_13__link', 'hand_link_5_0_link', 'hand_link_9_0_link', 'hand_link_2_0_link', 'hand_link_14__link', 'hand_link_6_0_link', 'hand_link_10__link', 'hand_link_3_0_link', 'hand_link_15__link', 'hand_link_7_0_link', 'hand_link_11__link', 'hand_link_3_0_link_tip_link', 'hand_link_15__link_tip_link', 'hand_link_7_0_link_tip_link', 'hand_link_11__link_tip_link']

# Arm links only -- everything up to and including end_effector_link, with the
# palm and all 20 finger links excluded. Used for the arm-to-arm collision
# sensor, which must NOT fire when the two hands meet (see
# update_collisions_bidirectional).
_GEN3_ARM_LINK_NAMES = [name for name in _GEN3_LINK_NAMES if not name.startswith("hand_")]
assert "forearm_link" in _GEN3_ARM_LINK_NAMES and "end_effector_link" in _GEN3_ARM_LINK_NAMES
assert not any(name.startswith("hand_") for name in _GEN3_ARM_LINK_NAMES)
assert len(_GEN3_ARM_LINK_NAMES) == 9


# Closed-grip hand pose, copied verbatim from UR5e_4f_CFG's init_state in
# robots_cfg.py -- the pose the BASELINE's holder uses to keep hold of the
# object for a whole episode.
#
# Diagnosed 2026-09-24: the two robot configs have OPPOSITE hand defaults.
# UR5e_4f_CFG curls three fingers plus the thumb (0.263*6 / *5 / *2.3), i.e. a
# closed grip. GEN3_4f_CFG sets only "joint_12_0": 0.263 (the thumb's zero) and
# is otherwise OPEN. Self-play replaced the UR5e with a GEN3 clone for the
# giving slot and carried over the arm but NOT the grip, so the giver has been
# presenting the object with an open hand in every run -- the object was never
# actually held. That is why it sank ~2.2mm/step with the arms motionless, and
# why the kinematic weld had to exist at all.
#
# Both arms carry the same 16-DOF hand (identical joint_0_0..joint_15_0 names),
# so these values transfer directly.
_CLOSED_HAND_JOINTS = {
    "joint_0_0": 0.0,    "joint_1_0": 0.263 * 6,  "joint_2_0": 0.263 * 5,  "joint_3_0": 0.263 * 2.3,
    "joint_4_0": 0.0,    "joint_5_0": 0.263 * 6,  "joint_6_0": 0.263 * 5,  "joint_7_0": 0.263 * 2.3,
    "joint_8_0": 0.0,    "joint_9_0": 0.263 * 6,  "joint_10_0": 0.263 * 5, "joint_11_0": 0.263 * 2.3,
    # joint_14_0 is NOT 0.263*5 like its neighbours. That value came straight
    # from UR5e_4f's config (robots_cfg.py:84), where the hand is a different
    # USD; on gen3_4f_arm2.usd this joint's range is roughly [-0.189, 0.800], so
    # 1.315 was out of it. IsaacLab's Articulation._validate_cfg raised
    # "joint_14_0: 1.315 not in [-0.189, 0.800]" on every startup -- and because
    # that raise happens inside an initialize CALLBACK, the exception was logged
    # and swallowed rather than stopping the run, so it sat unnoticed in the grip
    # verification (job 4288117) and in both self-play runs (4292718/19). PhysX
    # clamped the target to the limit anyway, which is why the grip still held.
    #
    # 0.79, not 0.80: _validate_cfg compares strictly (articulation.py:1260,
    # `default_joint_pos > joint_pos_limits[:, 1]`) and formats the limit to 3
    # decimals, so the printed 0.800 is a rounded value slightly BELOW 0.8 --
    # setting exactly 0.800 still failed (job 4292982, "0.800 not in [-0.189,
    # 0.800]"). A small margin keeps it inside whatever the true bound is; 0.01
    # rad on a thumb joint is ~0.6 degrees. Re-verified by job 4293464.
    "joint_12_0": 0.263, "joint_13_0": 0.0,       "joint_14_0": 0.79,      "joint_15_0": 0.263 * 5,
}


@configclass
class BimanualBidirectionalCfg(BimanualDirectCfg):

    # Recenter the render/video viewer camera on this task's own arm
    # midpoint. Diagnosed 2026-08-15: the parent's viewer (eye/lookat x
    # centered at -0.625) was tuned for the ORIGINAL single-arm slot layout
    # (GEN3 at -1.25, UR5e's original default at 0.0 -> midpoint -0.625).
    # Today's geometry fix moved the UR5e-slot clone to -0.65 (rotated to
    # face GEN3, closer together so the receiving arm's initial distance is
    # actually reachable -- see robot_cfg_1 below), shifting the true
    # midpoint to -0.95. The stale camera was pointed 0.325m away from where
    # the arms actually are, producing an edge-on, barely-legible framing
    # that looked like frozen arms in every rendered video even though
    # ground-truth joint telemetry showed both arms moving continuously
    # (confirmed via cumulative joint path length ~3.5-5.2 rad and ~20-25cm
    # net end-effector displacement over a full episode).
    viewer: ViewerCfg = ViewerCfg(
        # origin_type="env" is essential here. ViewerCfg defaults to
        # origin_type="world", meaning eye/lookat are ABSOLUTE world
        # coordinates -- but InteractiveScene lays environments out on a grid
        # whose origins depend on num_envs, so env 0's arms sit at world
        # coords like (7.6, -8.8) for a 64-env grid. A world-frame camera
        # aimed near the world origin was therefore pointing ~12m away from
        # the env it was meant to show, and the "arms" visible in every
        # rendered video all session were actually distant arms from OTHER
        # environments, at a range where this task's ~2cm-per-step motion is
        # sub-pixel -- which is exactly why they looked frozen while
        # joint_pos telemetry simultaneously proved continuous real motion.
        # Anchoring to env 0 makes the framing correct for any num_envs.
        origin_type="env",
        env_index=0,
        # Env-relative: GEN3 base at x=-1.25, UR5e-slot clone at x=-0.65,
        # so the pair straddles x=-0.95. Distances mirror the single-arm
        # baseline's known-good framing (1.3m out, 1.0m up).
        eye=(-0.95, 2.0, 1.5),
        lookat=(-0.95, 0.0, 1.0),
    )

    # ---- Action / observation sizing ----
    # 9 actions/arm (3 translation + 3 rotation + 3 hand control -- identical
    # per-arm meaning to BimanualDirectCfg's single-arm action). Action vector
    # is [UR5e (9) | GEN3 (9)], fixed order.
    num_actions = 9 * 2

    # Per-arm observation block: own ee pose rel. own base (7) + object/grasp
    # point pose rel. own base (7) + own selected hand-joint state (3) + other
    # arm's ee pose rel. own base (7) + signed role context (1, +1 if this arm
    # is currently receiving, -1 if giving/holding) = 25. Both blocks
    # concatenated in fixed order [UR5e, GEN3] -> 50. Changed 2026-08-13 from
    # a 2-dim one-hot [is_holder, is_receiver] to this single signed scalar
    # per direct instruction after a supervisor discussion. SharedArmActorCriticPolicy
    # reads arm_obs_dim from policy_kwargs in sb3_ppo_cfg_bidirectional.yaml --
    # that value and num_observations here must stay in sync with the block
    # layout in BimanualBidirectional._get_observations if it ever changes.
    arm_obs_dim = 25
    role_obs_dim = 1
    num_observations = arm_obs_dim * 2

    # ---- Phase-1 target ----
    # Pose the object should reach once GEN3 hands it back to UR5e (phase 1's
    # analogue of `target_pose`). Placeholder: UR5e's own initial end-effector
    # pose (ee_init_pose_quat[UR5e]) -- i.e. "bring it back to about where UR5e
    # started". NOT visually tuned in sim (this sandbox has no Isaac Sim/GPU to
    # check it against). Tune with debug_markers=True the same way the
    # existing `target_pose` was almost certainly tuned.
    target_pose_ur5e: torch.Tensor = torch.tensor([-0.5144, 0.1333, 0.6499, 0.2597, -0.6784, -0.2809, 0.6272])

    # Dual-quaternion translation-distance threshold for "object has arrived
    # at target_pose / target_pose_ur5e". Generalizes BimanualDirect's ad-hoc
    # `obj_pose[:, 0] < 0.05` check (bimanual_direct_env.py:649) -- that
    # heuristic was tuned to GEN3's specific approach frame and doesn't
    # transfer to UR5e's, so both phases use this symmetric, frame-
    # independent form instead (it's the alternative the original code's own
    # comment shows was considered). Not visually tuned in sim.
    obj_reach_target_thresh: float = 0.05

    # Episode needs room for two reach-and-grasp legs instead of one. Doubling
    # is a starting guess, not a measured value -- watch episode_length_buf /
    # outcome_success in the first real runs and adjust.
    episode_length_s = 6.0
    max_steps = 400

    # ---- Both arm slots use GEN3 (user-directed workaround, 2026-08-13) ----
    # UR5e's hand USD is missing the PhysX contact-reporter API on its finger
    # links, which crashes env.reset() before training can start (see
    # BimanualDirect/robots_cfg.py notes). Rather than wait on a proper asset
    # fix, spawn GEN3's already-working robot+hand model at the "UR5e" slot
    # too, on the assumption both arms are equivalent hardware for this run.
    # hand_joints and finger_tips (inherited from BimanualDirectCfg) already
    # use identical names for both slots -- only the arm-specific joint
    # names/ee_link/spawn pose differ by robot type, so only those are
    # overridden here.
    #
    # Spawn pose: reuse UR5e's ORIGINAL slot exactly (local pos=(0,0,0), no
    # rotation) rather than mirroring the clone to the opposite side. A first
    # attempt mirrored it to (1.25, 0, 0) with a 180 deg Z rotation, which
    # visually confirmed broken (2026-08-13, reset_video_out): that put the
    # two arms 2.5m apart (double the original 1.25m spacing) and the object
    # spawned at z=-0.19, below the ground plane -- the grasp-offset math
    # elsewhere in this file was tuned for UR5e's exact original slot and
    # does not generalize to an arbitrary mirrored pose. Explicitly zeroing
    # pos/rot here cancels out GEN3_4f_CFG's own baked-in pos=(-1.25,0,0),
    # since the object-spawn/grasp code assumes whatever occupies this slot
    # is where UR5e used to be.
    robot_cfg_1: Articulation = GEN3_4f_CFG.replace(
        prim_path="/World/envs/env_.*/UR5e",
        spawn=GEN3_4f_CFG.spawn.replace(
            usd_path=GEN3_4f_CFG.spawn.usd_path.replace("gen3_4f.usd", "gen3_4f_arm2.usd"),
        ),
        init_state=GEN3_4f_CFG.init_state.replace(
            pos=(-0.65, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),
            # Start the GIVING hand closed around the object (see
            # _CLOSED_HAND_JOINTS). Without this the giver presents the object
            # with an open palm and it is not held at all.
            joint_pos={**GEN3_4f_CFG.init_state.joint_pos, **_CLOSED_HAND_JOINTS},
        ),  # rotated 180deg about z to face GEN3. Separation 0.60m (GEN3 base -1.25, this arm -0.65). Reverted from 0.80m for Run A: 0.80m is confounded with the phase1_fraction 4.7-14% -> 0 regression, and 0.60m has a known-good baseline. Note the 'both reach the same point' worry that motivated 0.80m was based on the reset IK target (0.30m), NOT the reach limit -- and the measured hand separation never fell below 0.43m anyway, so contention was never the binding constraint.
    )
    # Both slots hold GEN3 hardware, so BOTH entries must use GEN3 link names.
    # Diagnosed 2026-08-15: this inherited BimanualDirectCfg.links unchanged,
    # whose [UR5e] entry lists real-UR5e names ('upper_arm_link',
    # 'wrist_1_link', 'camera_link', 'ee_link', ...) that do not exist on a
    # GEN3. Every contact sensor filtering on those paths silently matched
    # nothing -- the source of the "Filter pattern ... did not match the
    # correct number of entries (expected N, found 0)" spam in every log, and
    # the reason arm-to-arm collisions were effectively undetectable.
    links = [_GEN3_LINK_NAMES, _GEN3_LINK_NAMES]

    ee_link = ['tool_frame', 'tool_frame']
    joints = [
        ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5', 'arm_joint_6', 'arm_joint_7'],
        ['arm_joint_1', 'arm_joint_2', 'arm_joint_3', 'arm_joint_4', 'arm_joint_5', 'arm_joint_6', 'arm_joint_7'],
    ]
    hand_joints = [['joint_' + str(i) + '_0' for i in range(0, 16)] for _ in range(2)]
    all_joints = [joints[0] + hand_joints[0], joints[1] + hand_joints[1]]

    # BimanualDirectCfg.ee_init_pose_quat[UR5e] is a target end-effector pose
    # tuned for UR5e's own kinematics/reach envelope, sampled around at reset
    # to seed the IK controller (BimanualDirect.reset_robot_ee). Diagnosed
    # 2026-08-13: with the real UR5e target still in place, our GEN3 clone at
    # that slot cannot reach anywhere near it, so the IK solver was commanding
    # extreme joint velocities trying to satisfy an infeasible target --
    # ep_len_mean=1 and out_of_bounds~75% almost every step, immediately at
    # reset. Fix: reuse GEN3's own already-tuned target (row 1) for the clone
    # (row 0) instead. This is computed in each robot's own root-relative
    # frame (see BimanualDirect._get_ee_pose's subtract_frame_transforms), so
    # it is translation-invariant -- valid regardless of which world position
    # the clone's base sits at, as long as it is unrotated (it is, see
    # robot_cfg_1 above).
    ee_init_pose_quat = torch.tensor([
        [0.3046, 0.0250, 0.825, -0.6946, 0.2523, -0.6092, 0.2877],  # UR5e row: recomputed so world target = GEN3 row's world target, accounting for UR5e's new 180deg root rotation + 0.6m separation
        [0.2954, -0.0250, 0.825, -0.6946, 0.2523, -0.6092, 0.2877],
    ])
    _r, _p, _y = euler_xyz_from_quat(ee_init_pose_quat[:, 3:])
    _euler = torch.cat((_r.unsqueeze(-1), _p.unsqueeze(-1), _y.unsqueeze(-1)), dim=-1)
    ee_init_pose = torch.cat((ee_init_pose_quat[:, :3], _euler), dim=-1)

    # apply_range=[True, False] in the parent means only the "UR5e" slot gets
    # ee_pose_incs randomization added on top of its base target -- tuned for
    # what the real UR5e could tolerate. Our clone at that slot is GEN3, whose
    # base target we already fixed above, but the randomization magnitude is
    # a separate, still-UR5e-tuned parameter. Disabling randomization for both
    # slots (matching how the real GEN3 slot already resets deterministically)
    # removes this as a confound while diagnosing why the policy never once
    # completed phase 0 in a full 1250-iteration run despite std/entropy now
    # converging correctly (2026-08-14).
    # Penalty weights added 2026-08-15.
    # lambda_arm_collision: arm-to-arm contact was SENSED but weighted 0 in
    #   contact_matrix and never read in _get_dones, so driving the arms into
    #   each other was completely free. Penalised (not terminated) because
    #   over-strict conditions previously collapsed phase1_fraction to ~0.
    # lambda_holder_still: while the receiver is closing in, the holder should
    #   present the object steadily rather than moving it away -- a moving
    #   target makes the grasp far harder to learn. Gated on receiver
    #   proximity (exp(-k_decay*d)) so the holder is only constrained during
    #   the delicate final approach, and released once the receiver grasps.
    # Lowered 2.0 -> 0.5 alongside restricting the sensor to arm-only links
    # (2026-08-15). At 2.0 with the old hand-inclusive sensor this cost ~-86
    # per episode purely for bringing the hands together.
    lambda_arm_collision: float = 0.5
    # 3.0 -> 1.0 -> 0.0 (2026-08-15). Now redundant: stage_gate_arms freezes
    # the holder outright at stage 1, which enforces the same thing far more
    # directly. Measured at 0.04-0.08/step in runs 3908542/3908543 EVEN WITH
    # the freeze on (freezing the commanded increment leaves residual settling
    # velocity), making it a permanent unavoidable drain -- the main reason
    # per-step reward went net negative and the policy started truncating
    # episodes on purpose. Raise above 0 only if stage_gate_arms is disabled.
    lambda_holder_still: float = 0.0

    apply_range = [False, False]

    # Grasp-detection was firing on single-frame incidental contact (see
    # _phase_reward's contact_streak comment in bimanual_bidirectional_env.py,
    # diagnosed 2026-08-14). use_force_rewards switches contacts_flag from a
    # boolean any-touch count to an actual force-magnitude threshold (the
    # parent's F_threshold=0.10 was already configured but unused since
    # use_force_rewards defaulted False); grasp_sustain_steps additionally
    # requires that condition to hold for several consecutive steps before
    # latching new_receiver_grasped, since a one-off force spike from a fast
    # bump is still not the same thing as a held grasp.
    # Kinematic weld of the object to the current holder. Retained only as a
    # fallback: with _CLOSED_HAND_JOINTS the giver now holds the object with a
    # real grip, which is what makes "receiver still holds it after the giver
    # lets go" a meaningful test. While the weld is on, stage 3 -> 4 converts
    # 12/12 because the object physically cannot fall.
    # DEFAULT FLIPPED TO FALSE 2026-09-25, verified by job 4288117: with the
    # giver's hand actually closed, the object holds with the weld OFF -- it
    # drops 6mm over 300 steps with both arms commanded still (it used to sink
    # 242mm in 111 steps), and the giver registers real contact force of ~5.25
    # where it previously read exactly 0.0.
    #
    # This is what makes "receiver still holds it after the giver lets go" a
    # real test. With the weld on, stage 3 -> 4 converted 12/12 because the
    # object could not physically fall.
    # Sample which arm gives, per episode (2026-09-25). False reproduces the
    # old fixed assignment (the arm in the UR5e slot always gives), under which
    # the +/-1 context scalar was perfectly predictable from the observation
    # slot and so carried no information -- see BimanualBidirectional.__init__.
    randomize_giver: bool = True
    # Completed episodes per window for metrics/success_rate_per_episode.
    success_window: int = 500

    use_object_weld: bool = False

    use_force_rewards: bool = True
    grasp_sustain_steps: int = 3

    # ------------------------------------------------------------------
    # ALOHA-style staged single transfer (2026-08-15)
    # ------------------------------------------------------------------
    # Adopted from lerobot/act_aloha_sim_transfer_cube_human at the user's
    # request. Two structural borrowings from that task:
    #   1. The arms act in SEQUENCE -- the holder presents the object at a
    #      fixed meet pose and only then is the receiver allowed to move in.
    #      Previously both arms moved from step 0, so the receiver was chasing
    #      a target that was itself still moving.
    #   2. Success is ONE completed transfer, not a round trip. That is the
    #      published benchmark's own criterion (83% success with ACT); our
    #      round-trip definition was strictly harder and fired 0 times ever.
    # See BimanualBidirectional._get_rewards for the 0-4 ladder itself.

    # stage_gate_arms: freeze the receiver's Cartesian motion until the object
    #   has been presented, and freeze the holder's once the receiver has it.
    #   Set False to let both arms move freely throughout (old behaviour).
    stage_gate_arms: bool = True

    # Object must sit within handoff_thresh of the handoff point for
    # present_sustain_steps consecutive steps to count as "presented" --
    # a fly-through does not release the receiver.
    handoff_thresh: float = 0.08
    present_sustain_steps: int = 5
    # Steps the receiver must keep its grasp AFTER the holder has opened, for
    # the transfer to count. This is what distinguishes a real handover from a
    # momentary double grip.
    release_sustain_steps: int = 10
    # Handoff point is the midpoint of the two end effectors at reset, lifted
    # by this much so the meet pose is not down at table level.
    handoff_height_offset: float = 0.05

    # Potential-based shaping weights: reward is w * (prev_dist - dist) in
    # metres per step, clipped to shaping_clip metres to stop a single physics
    # glitch or teleport from paying out a huge one-step bonus. w=50 over a
    # ~0.5m approach totals ~25, deliberately smaller than the ladder bonuses
    # so shaping guides but does not dominate.
    w_present: float = 50.0
    w_reach: float = 50.0
    shaping_clip: float = 0.05

    # Sparse ladder bonuses, paid once each on entering the rung. Increasing so
    # the policy is always paid more for the next rung than for cycling on the
    # current one; a full success is ~375 including shaping, versus Run A's
    # flat ~79 that came almost entirely from an unconditional contact term.
    # Charged once when an episode ends via truncation (arm/object falling,
    # ground contact, velocity limit) rather than by timing out or succeeding.
    # Without this, ending the episode early is worth ~0 while playing it out
    # accumulates whatever the per-step reward happens to be -- so any net
    # negative per-step reward makes deliberate self-termination the optimal
    # policy, which is exactly what runs 3908542/3908543 converged to
    # (ep_len_mean 399 -> 1). Sized to exceed the worst plausible per-episode
    # shaping loss so surviving always dominates.
    penalty_truncation: float = 50.0

    bonus_present: float = 25.0
    bonus_touch: float = 50.0
    bonus_grasp: float = 100.0
    bonus_transfer: float = 200.0

    # ------------------------------------------------------------------
    # Reverse curriculum (2026-09-10, PI's suggestion)
    # ------------------------------------------------------------------
    # Standard reverse-curriculum idea (Florensa et al., CoRL 2017): start
    # episodes NEAR the goal so the policy succeeds immediately and has a
    # reward signal at all, then walk the start state backwards as it
    # improves.
    #
    # Motivated directly by the eval histogram of jobs 3908719/3908720: 66%
    # of episodes reach "hand touching the object" and only 0-3% reach "has a
    # grip". Approach is solved; the unsolved part is the single most
    # contact-rich instant, and under the normal reset the policy only gets
    # one attempt at that instant per episode -- after a long approach it
    # usually mistimes. Starting the receiver already at the object gives it
    # thousands of repetitions of the grip specifically.
    use_reverse_curriculum: bool = True

    # Fraction of the way from the receiver's sampled home pose to the
    # object's grasp point at episode start. 1.0 would be exactly on the
    # object; 0.0 is the normal reset (hardest, == the 3908719/3908720 runs).
    # The last entry MUST be 0.0 so the final level is the real task.
    curriculum_start_frac: list = [0.90, 0.70, 0.45, 0.20, 0.0]

    # Promotion/demotion on a rolling window of completed episodes. Demotion
    # matters as much as promotion: if a level turns out to be too hard the
    # run drops back rather than flatlining at 0 for the rest of training,
    # which is the failure mode every previous run had.
    # Iterations for the one-off IK solve that finds the "hand on the object"
    # joint configuration. A single differential step is nowhere near enough
    # for a ~0.3m target -- see _precompute_object_reach_q.
    curriculum_ik_iters: int = 40

    curriculum_window: int = 200
    # 0.35, lowered from 0.60 on 2026-10-02. Run 4295173 trained 409M steps and
    # NEVER left level 0 -- metrics/curriculum_level took exactly one value for
    # the whole run. Its pooled success rate peaked around 0.46, so a 0.60
    # promotion threshold was unreachable by construction and the "curriculum"
    # was just a permanently easier task: at frac 0.90 the meet point sits 0.019m
    # from the receiver's own hand (measured, job 4313021) and 0.362m from the
    # giver's, so the giver carries the object to the stationary receiver, which
    # travels 5-7mm and closes its fingers. Every number reported from that run
    # is a level-0 number. Same failure shape as the ICL demotion rule that
    # reversed every promotion: a rule that can never fire silently redefines
    # the task.
    curriculum_promote_rate: float = 0.35
    curriculum_demote_rate: float = 0.15
    # Belt-and-braces against the same class of stall: advance regardless of the
    # success rate after this many env-steps at one level, the way the ICL
    # curriculum does (icl_force_promote_steps). A performance-gated curriculum
    # that cannot reach its own gate makes no progress at all, so there has to
    # be a path upward that does not depend on the gate.
    curriculum_force_promote_steps: int = 50_000_000




def _mirror_finger_sensor(cfg, robot_idx: int, link_name: str, num_envs: int) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[robot_idx] + "/" + link_name,
        update_period=0.001,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[f"/World/envs/env_{i}/Cuboid" for i in range(num_envs)],
    )


def update_collisions_bidirectional(cfg, num_envs):
    '''
    Mirrors BimanualDirectCfg.update_collisions() for both arms instead of
    GEN3 only. Sensor dict insertion order fixes the column order of
    env.contacts / env.force_norms (see BimanualDirect.filter_collisions,
    inherited unchanged) -- contact_matrix below must match that order
    exactly: [ur5e_w_ground, gen3_w_ground,
              ur5e_fingers(14), ur5e_palm, ur5e_hand,
              gen3_fingers(14), gen3_palm, gen3_hand,
              robot1_w_robot2].
    '''

    contact_sensors_dict = {}

    # Ground contact is a failure condition for whichever arm is currently
    # driven by the policy, i.e. both now (not just GEN3).
    for robot_key in cfg.keys:
        contact_sensors_dict[f"{robot_key.lower()}_w_ground"] = ContactSensorCfg(
            prim_path="/World/envs/env_.*/" + robot_key + "/.*_link",
            update_period=0.001,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
        )

    for robot_idx, robot_key in enumerate(cfg.keys):
        for link in FINGER_LINKS:
            contact_sensors_dict[f"{robot_key.lower()}_{link}_w_object"] = _mirror_finger_sensor(
                cfg, robot_idx, link, num_envs
            )
        contact_sensors_dict[f"{robot_key.lower()}_palm_w_object"] = _mirror_finger_sensor(
            cfg, robot_idx, "palm_aux_.*", num_envs
        )
        contact_sensors_dict[f"{robot_key.lower()}_hand_w_object"] = _mirror_finger_sensor(
            cfg, robot_idx, "hand_link.*", num_envs
        )

    # Arm-to-arm collision -- one shared sensor, meaningful in either phase.
    # Arm-to-arm collision sensor. Restricted to the ARM links of both robots
    # (2026-08-15): it previously watched every `.*_link` on both sides, which
    # includes the palm and all 20 finger links, so the two HANDS meeting --
    # the entire point of a handover -- registered as an arm collision and was
    # charged lambda_arm_collision. Smoke job 3908485 measured 10.8% of steps
    # flagged, with the rate stepping up exactly as the arms closed in, i.e.
    # the penalty was actively fighting stage 2. Genuine forearm/upper-arm
    # collisions are still caught.
    arm_only = _GEN3_ARM_LINK_NAMES
    arm_link_regex = "(" + "|".join(arm_only) + ")"
    contact_sensors_dict["robot1_w_robot2"] = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.GEN3] + "/" + arm_link_regex,
        update_period=0.001,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            f"/World/envs/env_{i}/{cfg.keys[cfg.UR5e]}/{joint}"
            for i in range(num_envs)
            for joint in arm_only
        ],
    )

    cfg.contact_sensors_dict = contact_sensors_dict

    per_arm = torch.cat((FINGER_WEIGHTS, torch.tensor([PALM_WEIGHT, HAND_WEIGHT])))
    cfg.contact_matrix = torch.cat((
        torch.zeros(2),  # ur5e_w_ground, gen3_w_ground
        per_arm,          # ur5e fingers + palm + hand
        per_arm,          # gen3 fingers + palm + hand
        torch.zeros(1),   # robot1_w_robot2
    ))

    return cfg
