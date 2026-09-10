from __future__ import annotations

import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_from_euler_xyz, subtract_frame_transforms

from .bimanual_bidirectional_env_cfg import (
    ARM_CONTACT_WIDTH,
    FINGER_LINKS,
    BimanualBidirectionalCfg,
    update_collisions_bidirectional,
)
from .bimanual_direct_env import BimanualDirect
from .mdp.rewards import dual_quaternion_error

'''
                ############## IMPORTANT #################
Round trip: phase 0 is BimanualDirect's task unchanged (GEN3 receives the
object from UR5e); phase 1 mirrors it with roles swapped (UR5e receives it
back from GEN3). `self.phase` is a per-env tensor -- envs in a batch can be in
different phases simultaneously, so every phase-dependent computation below is
a masked tensor op, never a Python if on a whole-batch condition.

Shape contract with train/networks.py's SharedArmActorCriticPolicy: the
observation is [UR5e block (arm_obs_dim) | GEN3 block (arm_obs_dim)], each
block laid out as
  own ee pose rel. own base (7)
  + object/grasp-point pose rel. own base (7)
  + own selected hand-joint state (3)
  + other arm's ee pose rel. own base (7)
  + signed role scalar (+1 receiving / -1 giving) (1)
= 25 (BimanualBidirectionalCfg.arm_obs_dim). The action is
[UR5e action (9) | GEN3 action (9)], same per-arm meaning as BimanualDirect's
single 9-dim action. If either layout changes, update
BimanualBidirectionalCfg.arm_obs_dim/role_obs_dim and
sb3_ppo_cfg_bidirectional.yaml's policy_kwargs together.
'''

NUM_FINGER = len(FINGER_LINKS)
THUMB_SLICE = slice(NUM_FINGER - 5, NUM_FINGER)  # finger_4 (2 links) + thumb (3 links), mirrors BimanualDirect's `-8:-3` slice into its 18-wide global array
NON_THUMB_FINGER_GROUPS = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 11)]  # finger_1, finger_2, finger_3, finger_4 -- each its own group of sub-link sensors, for the multi-finger grasp check below
NUM_GLOBAL_CONTACTS = 2  # ur5e_w_ground, gen3_w_ground

REWARD_TERM_KEYS = {
    "approach", "target", "bonus", "signal_1_firmness", "signal_2_balance",
    "signal_instability", "velocity_penalty", "force_excess",
    "holder_still_penalty",
}


class BimanualBidirectional(BimanualDirect):
    cfg: BimanualBidirectionalCfg

    def __init__(self, cfg: BimanualBidirectionalCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # target_pose_ur5e isn't known to BimanualDirectCfg.update_cfg(), so
        # repeat/device-place it here the same way update_cfg() does for
        # target_pose.
        self.cfg.target_pose_ur5e = self.cfg.target_pose_ur5e.repeat(self.num_envs, 1).to(self.device)

        # 0 = GEN3 receiving from UR5e (BimanualDirect's task). 1 = UR5e
        # receiving from GEN3 (the "give back" leg).
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Phase-1 mirrors of the parent's obj_reached / prev_dist bookkeeping.
        self.ur5e_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.round_trip_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Consecutive-step counters gating grasp detection -- diagnosed
        # 2026-08-14: contacts_flag (see _phase_reward) fires True from a
        # single-frame incidental touch (self.contacts is boolean any-force,
        # and the boolean-count threshold only needs one non-thumb link plus
        # the thumb touching for one instant). Since new_receiver_grasped is
        # a one-way latch for the rest of the episode, one spurious brush
        # during approach permanently released both the object weld and the
        # holder's real finger grip, leaving the near-massless (0.00025kg)
        # object to drift under residual velocity/gravity with nothing
        # holding it -- visible in rendered video as the object floating and
        # sliding with no corresponding arm motion. Requiring the flag to
        # hold for grasp_sustain_steps consecutive steps (see cfg) before
        # latching filters out single-frame spikes.
        self.gen3_contact_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ur5e_contact_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_dist_ur5e = torch.full((self.num_envs,), torch.inf, device=self.device)
        self.prev_dist_target_ur5e = torch.full((self.num_envs,), torch.inf, device=self.device)

        # --- ALOHA-style staged single transfer (2026-08-15) ---------------
        # See _get_rewards' docstring. self.stage is a per-env one-way ratchet
        # 0..4 replacing the old two-phase round trip as the task definition.
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.present_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.receiver_grasp_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.release_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.transfer_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Handoff point: recomputed per episode in _reset_idx as the midpoint
        # between the two arms' end effectors, so it adapts to UR5e's
        # randomized start rather than being a hardcoded world coordinate.
        self.handoff_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # inf means "no previous measurement yet" -- the potential-based
        # shaping terms in _get_rewards skip the first step of each episode
        # rather than reading inf as an enormous improvement.
        self.prev_present_dist = torch.full((self.num_envs,), torch.inf, device=self.device)
        self.prev_reach_dist = torch.full((self.num_envs,), torch.inf, device=self.device)
        # Set by _get_dones, consumed by _get_rewards (dones runs first in
        # DirectRLEnv.step). Initialised here so the very first _get_rewards
        # call is safe if the order ever changes.
        self._last_truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- Reverse curriculum state (2026-09-10) -------------------------
        # One global level shared by all envs (not per-env): promotion is
        # decided from a pooled success rate, so a per-env level would just
        # add variance without adding signal.
        self.curriculum_level = 0
        self._curr_ep_count = 0
        self._curr_succ_count = 0

        # Per-arm pose bookkeeping filled in by update_new_poses(); replaces
        # the parent's GEN3-only singular tensors of the same name/purpose.
        self.rot_ee_pose_r = [torch.zeros((self.num_envs, 7), device=self.device) for _ in self.cfg.keys]
        self.tips_pose_r = [torch.zeros((self.num_envs, 7), device=self.device) for _ in self.cfg.keys]
        self.tips_pose_r_back = [torch.zeros((self.num_envs, 7), device=self.device) for _ in self.cfg.keys]
        self.grasp_point_obj_pose_r = [torch.zeros((self.num_envs, 7), device=self.device) for _ in self.cfg.keys]

        # Contact sensor columns are [ur5e_w_ground, gen3_w_ground,
        # ur5e_block(ARM_CONTACT_WIDTH), gen3_block(ARM_CONTACT_WIDTH),
        # robot1_w_robot2] -- see update_collisions_bidirectional(). Slice
        # once here rather than recomputing every reward step.
        self._ur5e_contact_slice = slice(NUM_GLOBAL_CONTACTS, NUM_GLOBAL_CONTACTS + ARM_CONTACT_WIDTH)
        self._gen3_contact_slice = slice(
            NUM_GLOBAL_CONTACTS + ARM_CONTACT_WIDTH, NUM_GLOBAL_CONTACTS + 2 * ARM_CONTACT_WIDTH
        )

        # NOTE (v1 scope cuts, not wired up for this env): debug marker
        # visualization (update_markers(), inherited from BimanualDirect)
        # still reads the parent's GEN3-only debug_* buffers, which
        # update_new_poses() below no longer populates -- leave
        # cfg.debug_markers False. Camera rendering (render_imgs/save_imgs)
        # is likewise not reproduced here.

    # Only line that differs from BimanualDirect._setup_scene: mirrored
    # contact sensors for both arms instead of GEN3 only.
    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations[self.cfg.keys[self.cfg.UR5e]] = Articulation(self.cfg.robot_cfg_1)
        self.scene.articulations[self.cfg.keys[self.cfg.GEN3]] = Articulation(self.cfg.robot_cfg_2)

        self.cfg = update_collisions_bidirectional(self.cfg, num_envs=self.num_envs)
        for idx, sensor_cfg in self.cfg.contact_sensors_dict.items():
            self.scene.sensors[idx] = ContactSensor(sensor_cfg)

        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)
        self.scene.extras["markers"] = VisualizationMarkers(self.cfg.marker_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # Identical to BimanualDirect._preprocess_actions except for the removed
    # "open UR5e's hand once GEN3 has grasped" side effect (bimanual_direct_
    # env.py:258), which was hardcoded to one arm/one direction -- both
    # directions' releases are handled explicitly by _apply_releases() below
    # instead, since both arms are policy-driven here.
    def _preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions, -1, 1)
        actions[:, :3] *= self.cfg.translation_scale

        actions_quat = torch.zeros((self.num_envs, 7 + 16)).to(self.device)
        actions_quat[:, :3] = actions[:, :3]

        hand_joint_index = 6 + int(not self.cfg.euler_flag)
        val = (actions[:, hand_joint_index:] * self.cfg.hand_joint_scale).repeat_interleave(4, dim=-1)

        actions_quat[:, 7] = 0
        actions_quat[:, 8] = val[:, 0] * 2
        actions_quat[:, 9] = 0
        actions_quat[:, 10] = 0
        actions_quat[:, 11] = val[:, 1] * 5
        actions_quat[:, 12] = 0.0
        actions_quat[:, 13] = val[:, 2] * 5
        actions_quat[:, 14] = val[:, 3] * 5

        actions_quat[:, 15] = val[:, 4] * 4
        actions_quat[:, 16] = val[:, 5] * 3
        actions_quat[:, 17] = val[:, 6] * 4
        actions_quat[:, 18] = val[:, 7] * 4

        actions_quat[:, 19] = val[:, 8] * 2.5
        actions_quat[:, 20] = val[:, 9] * 2.5
        actions_quat[:, 21] = val[:, 10] * 2.5
        actions_quat[:, 22] = val[:, 11] * 2.5

        if self.cfg.euler_flag:
            actions[:, 3:6] *= self.cfg.angle_scale
            actions_quat[:, 3:7] = quat_from_euler_xyz(roll=actions[:, 3], pitch=actions[:, 4], yaw=actions[:, 5])
        else:
            actions_quat[:, 3] *= self.cfg.angle_scale
            actions_quat[:, 4:7] = torch.nn.functional.normalize(actions_quat[:, 4:7])
            w = torch.cos(actions_quat[:, 3] / 2).unsqueeze(dim=0).T
            v = actions_quat[:, 4:7]
            sin_a = torch.sin(actions_quat[:, 3] / 2).unsqueeze(dim=0).T
            q = sin_a * v
            actions_quat[:, 3:7] = torch.cat((w, q), dim=1)

        return actions_quat

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions_ur5e = self._preprocess_actions(actions[:, :9])
        actions_gen3 = self._preprocess_actions(actions[:, 9:])

        self._gate_arm_motion(actions_ur5e, actions_gen3)

        self.perform_increment(idx=self.cfg.UR5e, actions=actions_ur5e)
        self.perform_increment(idx=self.cfg.GEN3, actions=actions_gen3)

        self._apply_releases()
        self._weld_object_to_current_holder()

    @staticmethod
    def _freeze_pose_increment(actions_quat: torch.Tensor, mask: torch.Tensor):
        '''
        Neutralize the per-step Cartesian increment for the masked envs:
        zero translation and identity rotation, so perform_increment leaves
        the IK target exactly where it was. Columns 7: (the 16 hand joints)
        are deliberately left untouched -- a waiting arm is still allowed to
        pre-shape/open its hand, it just may not travel.
        '''
        if not torch.any(mask):
            return
        actions_quat[mask, 0:3] = 0.0
        actions_quat[mask, 3] = 1.0
        actions_quat[mask, 4:7] = 0.0

    def _gate_arm_motion(self, actions_ur5e: torch.Tensor, actions_gen3: torch.Tensor):
        '''
        ALOHA-style temporal separation of the two arms (2026-08-15, at the
        user's request after reviewing lerobot/act_aloha_sim_transfer_cube_human).
        In that task the two arms never negotiate simultaneously: the first arm
        picks the cube and brings it to a fixed meet pose, and only then does
        the second arm move in to take it. Ours previously had both arms free
        to move from step 0, which meant the "handoff pose" was a moving target
        the receiver could never converge on -- consistent with Run A's receiver
        closing to 0.317m and then plateauing there for the rest of training.

        Stage 0 (holder still presenting): the receiver's arm is frozen.
        Stage >=1 (object is at the handoff point): the HOLDER's arm freezes,
        for the rest of the episode. In ALOHA the presenting arm parks at the
        meet pose and simply waits; the receiver approaches a stationary
        target. Its hand joints stay free so it can still open to release.

        Freezing the holder at stage >=1 rather than >=3 also removes a
        perverse incentive found in smoke job 3908485: holder_still_penalty
        applies during stages 1-2, the ladder never ratchets backwards, and
        stage 2->3 rarely fires -- so an env that reached stage 1 paid that
        penalty for the whole remaining episode (measured 0.988/step, ~-395
        per episode, against a total ladder value of 375). The cheapest
        available policy was therefore to NEVER present the object and sit in
        stage 0 forever. With the holder frozen the penalty is ~0 by
        construction and cannot be dodged by refusing to progress.
        '''
        if not self.cfg.stage_gate_arms:
            return
        self._freeze_pose_increment(actions_gen3, self.stage == 0)
        self._freeze_pose_increment(actions_ur5e, self.stage >= 1)

    def _weld_object_to_current_holder(self):
        '''
        Object is never a free-falling projectile and never fully released
        to raw physics mid-episode: it stays rigidly attached (kinematically
        re-pinned each step, same reset-time hand-relative offset) to
        whichever hand currently "has" it per the grasp state machine --
        the phase's holder before its receiver has grasped, the receiver
        itself afterward. Both arms are the same physical hand geometry, so
        the same offset is valid for either.

        Diagnosed 2026-08-14 in two stages: (1) originally the object was
        only positioned once at reset with no ongoing support, so it
        free-fell under gravity from step 0 and hit object_height_limit
        (failure) in ~27 steps before any grasp had ever occurred
        (metrics/contact_forces was 0.0 across every step of every rollout
        inspected) -- fixed by welding to the holder pre-grasp. (2) that
        first fix released the weld entirely to free physics once
        new_receiver_grasped latched, but that latch is a ONE-WAY switch for
        the rest of the episode, and grasp-detection -- even after adding a
        real force-magnitude threshold, a >=2-distinct-finger-groups
        requirement, and a multi-step sustain counter -- can still fire once
        from a brief incidental sweep-through during fast exploratory
        motion. A single such event permanently left the near-massless
        (0.00025kg) object with nothing physically holding it for the rest
        of the episode, which is what produced the "floating and drifting
        independent of visible arm motion" behaviour seen in rendered video.
        Continuing to weld to the receiver post-grasp (instead of releasing)
        makes a still-imperfect detection harmless -- the object just
        kinematically follows the "wrong" hand -- rather than catastrophic.
        '''
        # Single-transfer staging (2026-08-15): UR5e holds until the receiver
        # has a sustained grasp (stage >= 3), GEN3 holds from then on.
        gen3_is_holder = self.stage >= 3

        for idx, holder_mask in (
            (self.cfg.UR5e, torch.logical_not(gen3_is_holder)),
            (self.cfg.GEN3, gen3_is_holder),
        ):
            if not torch.any(holder_mask):
                continue
            env_ids = holder_mask.nonzero(as_tuple=False).squeeze(-1)
            key = self.cfg.keys[idx]
            ee_pose_w = self.scene.articulations[key].data.body_state_w[env_ids, self.ee_jacobi_idx[idx] + 1, 0:7]
            obj_pos, obj_quat = combine_frame_transforms(
                t01=ee_pose_w[:, :3], q01=ee_pose_w[:, 3:],
                t12=self.cfg.obj_pos_trans[env_ids], q12=self.cfg.obj_quat_trans[env_ids],
            )
            self.scene.rigid_objects["object"].write_root_pose_to_sim(
                root_pose=torch.cat((obj_pos, obj_quat), dim=-1), env_ids=env_ids
            )
            # RESTORED 2026-08-15 (job 3908621). This call was removed earlier
            # today on the theory that it broke the render pipeline; the code
            # comment here claimed that was "PROVEN via isolation testing". It
            # was not -- the actual cause of the frozen-looking video was two
            # prims sharing one USD prototype (fixed separately via
            # gen3_4f_arm2.usd), which was only identified afterwards.
            #
            # Removing it silently broke the weld. Writing the pose without
            # also zeroing the velocity leaves the object's accumulated
            # downward velocity intact, so physics drags it below the written
            # pose every substep. Measured with the arms commanded to hold
            # perfectly still (ur5e_ee_z constant at 0.790 for 120 steps): the
            # object sank linearly 0.944 -> 0.702, ~2.2mm/step, reaching
            # object_height_limit (0.2499) around step ~318 and truncating the
            # episode regardless of what the policy did. That -- not any
            # learned behaviour -- is what drove ep_len_mean from 399 to 1 in
            # runs 3908542/3908543/3908605/3908606.
            self.scene.rigid_objects["object"].write_root_velocity_to_sim(
                root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
            )

    def _apply_releases(self):
        '''
        Mirrors BimanualDirect's "holder opens its hand once the receiver
        grasps" behaviour (bimanual_direct_env.py:258) in both directions:
        UR5e opens once GEN3 grasps (phase 0, same as the original task),
        GEN3 opens once UR5e grasps (phase 1). Hand joints are always the
        last 16 columns of self.actions[idx] regardless of arm DOF (6 for
        UR5e, 7 for GEN3), so -16: is used rather than a fixed offset.
        '''
        release = self.stage >= 3
        self.actions[self.cfg.UR5e][release, -16:] = self.open_hand_joints

    def _apply_action(self) -> None:
        self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].set_joint_position_target(
            self.actions[self.cfg.UR5e], joint_ids=self._all_joints_idx[self.cfg.UR5e]
        )
        self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].set_joint_position_target(
            self.actions[self.cfg.GEN3], joint_ids=self._all_joints_idx[self.cfg.GEN3]
        )

    def update_new_poses(self):
        for idx in (self.cfg.UR5e, self.cfg.GEN3):
            ee_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.body_state_w[:, self.ee_jacobi_idx[idx] + 1, 0:7]
            root_pose_w = self.scene.articulations[self.cfg.keys[idx]].data.root_state_w[:, 0:7]

            rot_ee_pos_r, rot_ee_quat_r = subtract_frame_transforms(
                t01=root_pose_w[:, :3], q01=root_pose_w[:, 3:], t02=ee_pose_w[:, :3], q02=ee_pose_w[:, 3:]
            )
            self.rot_ee_pose_r[idx] = torch.cat((rot_ee_pos_r, rot_ee_quat_r), dim=-1)

            tips_pose_w = torch.mean(
                self.scene.articulations[self.cfg.keys[idx]].data.body_state_w[:, self.finger_tips[idx], 0:7], dim=-2
            )
            tip_pos_r, tip_or_r = subtract_frame_transforms(
                t01=root_pose_w[:, :3], q01=root_pose_w[:, 3:], t02=tips_pose_w[:, :3], q02=tips_pose_w[:, 3:]
            )
            tips_pose_r = torch.cat((tip_pos_r, tip_or_r), dim=-1)
            tips_pose_r[:, 3:] = self.rot_ee_pose_r[idx][:, 3:]
            self.tips_pose_r_back[idx] = tips_pose_r.clone()

            tip_pos_r, tip_or_r = combine_frame_transforms(
                t01=tips_pose_r[:, :3],
                q01=tips_pose_r[:, 3:],
                t12=self.cfg.tips_displacement,
                q12=torch.tensor([1, 0, 0, 0]).to(self.device).repeat(self.num_envs, 1),
            )
            self.tips_pose_r[idx] = torch.cat((tip_pos_r, tip_or_r), dim=-1)

            obj_pose_w = self.scene.rigid_objects["object"].data.body_state_w[:, 0, :7]
            grasp_point_obj_pos_w, grasp_point_obj_quat_w = combine_frame_transforms(
                t01=obj_pose_w[:, :3],
                q01=obj_pose_w[:, 3:],
                t12=self.cfg.grasp_obs_obj_pos_trans,
                q12=self.cfg.grasp_obs_obj_quat_trans,
            )
            # Same object-frame-to-grasp-point offset reused for both arms
            # (identical hand hardware) -- not visually verified for UR5e's
            # approach angle, unlike GEN3's which the original task tuned.
            grasp_point_obj_pos_w, grasp_point_obj_quat_w = combine_frame_transforms(
                t01=grasp_point_obj_pos_w,
                q01=grasp_point_obj_quat_w,
                t12=torch.zeros_like(grasp_point_obj_pos_w),
                q12=self.cfg.rot_225_z_pos_quat,
            )
            grasp_point_obj_pos_r, grasp_point_obj_quat_r = subtract_frame_transforms(
                t01=root_pose_w[:, :3], q01=root_pose_w[:, 3:], t02=grasp_point_obj_pos_w, q02=grasp_point_obj_quat_w
            )
            self.grasp_point_obj_pose_r[idx] = torch.cat((grasp_point_obj_pos_r, grasp_point_obj_quat_r), dim=-1)

    def _selected_hand_joint(self, idx: int) -> torch.Tensor:
        hand_joint_pos = self.scene.articulations[self.cfg.keys[idx]].data.joint_pos[:, self._hand_joints_idx[idx]]
        sel = torch.round(
            torch.cat((hand_joint_pos[:, 1].unsqueeze(-1), hand_joint_pos[:, 4].unsqueeze(-1), hand_joint_pos[:, 6:]), dim=-1),
            decimals=1,
        )
        return sel.view(-1, 3, 4).mean(dim=-1).view(-1, 3)

    def _other_arm_ee_in_own_frame(self, self_idx: int, other_idx: int) -> torch.Tensor:
        other_ee_pose_w = self.scene.articulations[self.cfg.keys[other_idx]].data.body_state_w[:, self.ee_jacobi_idx[other_idx] + 1, 0:7]
        self_root_pose_w = self.scene.articulations[self.cfg.keys[self_idx]].data.root_state_w[:, 0:7]
        pos_r, quat_r = subtract_frame_transforms(
            t01=self_root_pose_w[:, :3], q01=self_root_pose_w[:, 3:], t02=other_ee_pose_w[:, :3], q02=other_ee_pose_w[:, 3:]
        )
        return torch.cat((pos_r, quat_r), dim=-1)

    def _get_observations(self) -> dict:
        self.filter_collisions()
        self.update_new_poses()

        # Signed context scalar (+1 receiving, -1 giving/holding) per user
        # direction after a supervisor discussion, 2026-08-13 -- replaces the
        # earlier 2-dim one-hot [is_holder, is_receiver]. Under the staged
        # single transfer (2026-08-15) GEN3 is the receiver until it has the
        # object (stage >= 3), after which the roles flip -- so the scalar is
        # still the thing that tells the shared policy which arm it is driving.
        gen3_has_object = (self.stage >= 3).float()
        role_ur5e = (2.0 * gen3_has_object - 1.0).unsqueeze(-1)
        role_gen3 = (1.0 - 2.0 * gen3_has_object).unsqueeze(-1)

        ur5e_block = torch.cat((
            self.rot_ee_pose_r[self.cfg.UR5e],
            self.grasp_point_obj_pose_r[self.cfg.UR5e],
            self._selected_hand_joint(self.cfg.UR5e),
            self._other_arm_ee_in_own_frame(self.cfg.UR5e, self.cfg.GEN3),
            role_ur5e,
        ), dim=-1)

        gen3_block = torch.cat((
            self.rot_ee_pose_r[self.cfg.GEN3],
            self.grasp_point_obj_pose_r[self.cfg.GEN3],
            self._selected_hand_joint(self.cfg.GEN3),
            self._other_arm_ee_in_own_frame(self.cfg.GEN3, self.cfg.UR5e),
            role_gen3,
        ), dim=-1)

        obs = torch.cat((ur5e_block, gen3_block), dim=-1)

        return {"policy": obs, "phase": self.phase.clone()}

    def _phase_reward(
        self, receiver_idx, holder_idx, contacts_local, force_norms_local,
        prev_contacts_local, prev_force_norms_local, arm_contact_matrix,
        target_pose, prev_dist, prev_dist_target, receiver_grasped,
        contact_streak,
    ):
        '''
        Reusable grasp-approach-then-transport reward, parameterized by which
        arm is receiving vs. holding. Mirrors BimanualDirect._get_rewards
        (bimanual_direct_env.py:570-773) term-for-term, adapted from that
        method's hardcoded 18-wide global contact array + fixed GEN3 identity
        to this env's ARM_CONTACT_WIDTH-wide per-arm slices + parameterized
        receiver/holder. See THUMB_SLICE's docstring for how the index
        remapping was derived.

        Returns (reward, hand_obj_dist, obj_target_dist, new_receiver_grasped,
        just_transitioned, reached_target, log, new_streak).
        '''
        cfg = self.cfg
        device = self.device

        ee_pose = self.tips_pose_r[receiver_idx]
        ee_pose_back = self.tips_pose_r_back[receiver_idx]
        obj_pose = self.grasp_point_obj_pose_r[receiver_idx]

        tips_receiver = self.scene.articulations[cfg.keys[receiver_idx]].data.body_state_w[:, self.finger_tips[receiver_idx], 0:7].mean(1)[:, :3]
        tips_holder = self.scene.articulations[cfg.keys[holder_idx]].data.body_state_w[:, self.finger_tips[holder_idx], 0:7].mean(1)[:, :3]
        obj = self.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]

        hand_obj_dist = dual_quaternion_error(ee_pose, obj_pose, device)
        hand_obj_dist_back = dual_quaternion_error(ee_pose_back, obj_pose, device)
        obj_target_dist = dual_quaternion_error(obj_pose, target_pose, device)

        tips_dist = torch.norm((tips_holder - tips_receiver), dim=-1)
        obj_dist = torch.norm((obj - tips_receiver), dim=-1)

        contacts_w = contacts_local * arm_contact_matrix
        thumb_w = contacts_w[:, THUMB_SLICE].clone()
        thumb_con = thumb_w.sum(-1) > 0.0

        if cfg.use_force_rewards and cfg.F_threshold > 0:
            force_normalized = torch.tanh(force_norms_local / cfg.F_ref)
            force_w_trans = force_normalized * arm_contact_matrix
            thumb_force_w = force_w_trans[:, THUMB_SLICE].clone()
            contacts_flag = torch.logical_and(
                force_w_trans[:, :-1].sum(-1) - thumb_force_w.sum(-1) > cfg.F_threshold,
                thumb_force_w.sum(-1) > 0.0,
            )
        else:
            contacts_flag = torch.logical_and(contacts_w[:, :-1].sum(-1) - thumb_w.sum(-1) > 0.4, thumb_con)

        # Multi-finger requirement -- diagnosed 2026-08-14: the threshold
        # check above sums ALL of finger_1/2/3's own sub-link sensors
        # together (FINGER_LINKS packs 2-3 contact-sensor links per physical
        # finger), so a SINGLE finger resting/pressing against the object
        # with its own multiple sub-links active can clear the threshold
        # alone (one fingertip alone is weight 0.65 > 0.4). That let
        # obj_reached/ur5e_grasped latch permanently from one finger's
        # sustained contact, not a real multi-finger grasp -- releasing both
        # the kinematic weld and the holder's real grip with nothing left to
        # mechanically support the object. Require at least 2 of the 4
        # non-thumb finger groups (not sub-links) plus the thumb.
        finger_group_touch = torch.stack(
            [(contacts_w[:, g] > 0.0).any(dim=-1) for g in NON_THUMB_FINGER_GROUPS], dim=-1
        )
        multi_finger_grasp = torch.logical_and(finger_group_touch.float().sum(-1) >= 2, thumb_con)
        contacts_flag = torch.logical_and(contacts_flag, multi_finger_grasp)

        new_streak = torch.where(contacts_flag, contact_streak + 1, torch.zeros_like(contact_streak))
        sustained_flag = new_streak >= cfg.grasp_sustain_steps

        just_transitioned = torch.logical_and(sustained_flag, torch.logical_not(receiver_grasped))
        bonus = receiver_grasped.clone()
        new_receiver_grasped = torch.logical_or(sustained_flag, receiver_grasped)
        bonus = torch.logical_and(new_receiver_grasped.clone(), torch.logical_not(bonus))

        # Generalized "object arrived at target" check: original task used an
        # ad-hoc `obj_pose[:, 0] < 0.05` (a heuristic tuned to GEN3's specific
        # approach frame -- see bimanual_direct_env.py:649, whose own comment
        # shows this dual-quaternion-distance form as the considered
        # alternative). That heuristic doesn't transfer to UR5e's differently
        # oriented approach, so both phases use this symmetric form instead.
        reached_target = obj_target_dist[:, 1] < cfg.obj_reach_target_thresh

        # RUN-A CHANGE 2026-08-15: shape approach/transport on the TRANSLATION
        # module (index 1, metres) instead of the dual-quaternion module
        # (index 0, which blends position AND rotation).
        # Measured on job 3905946's checkpoint: the receiver's fingertips went
        # 0.4464m -> 0.4734m over 300 steps (moved AWAY, never below 0.435m)
        # while the logged dual-quat "hand_obj_dist" fell to 0.379 and the
        # sparkline trended down -- i.e. the shaped quantity improved while
        # actual Euclidean approach did not. A rotation-blended metric can be
        # minimised by spinning the wrist, and arm_joint_7 (wrist roll) has
        # shown the largest travel of any joint in every run. Index 1 is
        # metres, so "closer" can only mean closer.
        dist = hand_obj_dist[:, 1] * torch.logical_not(new_receiver_grasped).int() + obj_target_dist[:, 1] * new_receiver_grasped.int()
        prev_dist_eff = prev_dist * torch.logical_not(new_receiver_grasped).int() + prev_dist_target * new_receiver_grasped.int()

        pre_mod = torch.logical_and(tips_dist > obj_dist, hand_obj_dist_back[:, 1] > hand_obj_dist[:, 1])
        mod = torch.logical_and(dist < prev_dist_eff, pre_mod).float()

        rew_scale_hand_obj = cfg.rew_scale_hand_obj * torch.ones(self.num_envs, device=device)
        rew_scale_hand_obj = rew_scale_hand_obj / (contacts_local[:, :-1].sum(-1) + 1)

        reward_1 = mod * rew_scale_hand_obj * torch.exp(-2 * hand_obj_dist[:, 1]) / (1 + 2 * (torch.logical_not(pre_mod)).int())
        reward_2 = cfg.rew_scale_obj_target * torch.exp(-2 * obj_target_dist[:, 1])

        reward = reward_1 * torch.logical_not(new_receiver_grasped) + 10 * reward_2 * new_receiver_grasped + cfg.bonus_obj_reach * bonus / 2
        reward = reward + contacts_w[:, :-1].sum(-1)

        log = {"approach": reward_1, "target": reward_2}

        if cfg.use_signal_1:
            force_normalized = torch.tanh(force_norms_local / cfg.F_ref)
            force_w = force_normalized * arm_contact_matrix
            signal_1_value = force_w[:, :-1].sum(-1)
            reward = reward + signal_1_value
            log["signal_1_firmness"] = signal_1_value

        if cfg.use_signal_2:
            force_normalized = torch.tanh(force_norms_local / cfg.F_ref)
            force_w = force_normalized * arm_contact_matrix
            thumb_force = force_w[:, THUMB_SLICE].sum(-1)
            finger_force = force_w[:, 0 : NUM_FINGER - 5].sum(-1) + cfg.palm_weight * force_w[:, NUM_FINGER : NUM_FINGER + 1].sum(-1)
            balance_imbalance = torch.abs(thumb_force - finger_force) / (thumb_force + finger_force + 1e-6)
            total_grasp_force = force_w[:, 0:NUM_FINGER].sum(-1)
            balance_active = (total_grasp_force > cfg.F_threshold).float()
            r_balance = -cfg.lambda_balance * balance_imbalance * new_receiver_grasped.float() * balance_active
            reward = reward + r_balance
            log["signal_2_balance"] = r_balance

        if cfg.use_signal_instability:
            pattern_change = (contacts_local[:, :-1].float() - prev_contacts_local[:, :-1]).abs().sum(-1)
            current_total = force_norms_local[:, :-1].sum(-1)
            prev_total = prev_force_norms_local[:, :-1].sum(-1)
            force_drop = torch.relu(prev_total - current_total)
            r_instability = -cfg.lambda_instability * pattern_change * force_drop * new_receiver_grasped.float()
            reward = reward + r_instability
            log["signal_instability"] = r_instability

        bonus_term = cfg.bonus_obj_reach * reached_target.float() * (contacts_w[:, :-1].sum(-1) > 0.0).float() * new_receiver_grasped.float()
        reward = reward + bonus_term
        log["bonus"] = bonus_term

        ee_vel_norm = torch.norm(
            self.scene.articulations[cfg.keys[receiver_idx]].data.body_state_w[:, self.ee_jacobi_idx[receiver_idx] + 1, 7:10], dim=-1
        )
        if cfg.vel_rew:
            excess_vel = torch.nn.functional.relu(ee_vel_norm - cfg.v_min)
            velocity_penalty = cfg.lambda_vel * (excess_vel**2) * torch.exp(-cfg.k_decay * hand_obj_dist[:, 1]) * torch.logical_not(new_receiver_grasped).float()
            reward = reward - velocity_penalty
            log["velocity_penalty"] = velocity_penalty

        # Holder stability: while the receiver is still closing in, the arm
        # currently holding the object should keep it as still as possible.
        # Gated by exp(-k_decay*hand_obj_dist) so it only bites during the
        # final approach, and switched off once the receiver has grasped
        # (after which the holder must move to release).
        holder_ee_vel = self.scene.articulations[cfg.keys[holder_idx]].data.body_state_w[
            :, self.ee_jacobi_idx[holder_idx] + 1, 7:
        ]
        holder_speed = torch.norm(holder_ee_vel, dim=-1)
        holder_still_penalty = (
            cfg.lambda_holder_still
            * holder_speed
            * torch.exp(-cfg.k_decay * hand_obj_dist[:, 1])
            * torch.logical_not(new_receiver_grasped).float()
        )
        reward = reward - holder_still_penalty
        log["holder_still_penalty"] = holder_still_penalty

        if cfg.use_force_rewards:
            excess_force = torch.relu(force_norms_local[:, :-1] - cfg.F_safe)
            r_excess = -cfg.lambda_force_excess * excess_force.sum(-1)
            reward = reward + r_excess
            log["force_excess"] = r_excess

        log["hand_obj_dist"] = hand_obj_dist[:, 1]        # metres (was dual-quat)
        log["obj_target_dist"] = obj_target_dist[:, 1]      # metres (was dual-quat)
        log["ee_vel_norm"] = ee_vel_norm
        log["contact_forces"] = contacts_w.sum(-1)
        log["force_at_transition"] = (
            force_norms_local[:, :NUM_FINGER][just_transitioned].sum(-1).mean().item() if just_transitioned.any() else 0.0
        )
        log["velocity_at_transition"] = ee_vel_norm[just_transitioned].mean().item() if just_transitioned.any() else 0.0

        return reward, hand_obj_dist[:, 1], obj_target_dist[:, 1], new_receiver_grasped, just_transitioned, reached_target, log, new_streak

    def _log_phase(self, log: dict, suffix: str):
        if "log" not in self.extras:
            self.extras["log"] = dict()
        for key, value in log.items():
            val = value.mean().item() if torch.is_tensor(value) else value
            prefix = "reward_terms" if key in REWARD_TERM_KEYS else "metrics"
            self.extras["log"][f"{prefix}/{key}{suffix}"] = val

    def _grasp_flags(self, idx: int, arm_contact_matrix: torch.Tensor):
        '''
        (touch, grasp) for one arm. `touch` is any object contact on the hand;
        `grasp` is the hardened multi-finger criterion carried over from
        _phase_reward -- force above F_threshold, thumb engaged, and at least 2
        of the 4 non-thumb finger GROUPS (not sub-links) in contact. See
        _phase_reward's comments for why the group requirement exists.
        '''
        cfg = self.cfg
        sl = self._gen3_contact_slice if idx == cfg.GEN3 else self._ur5e_contact_slice
        contacts_w = self.contacts[:, sl] * arm_contact_matrix
        thumb_w = contacts_w[:, THUMB_SLICE]
        thumb_con = thumb_w.sum(-1) > 0.0
        touch = contacts_w[:, :-1].sum(-1) > 0.0

        if cfg.use_force_rewards and cfg.F_threshold > 0:
            force_w = torch.tanh(self.force_norms[:, sl] / cfg.F_ref) * arm_contact_matrix
            firm = torch.logical_and(
                force_w[:, :-1].sum(-1) - force_w[:, THUMB_SLICE].sum(-1) > cfg.F_threshold,
                force_w[:, THUMB_SLICE].sum(-1) > 0.0,
            )
        else:
            firm = torch.logical_and(contacts_w[:, :-1].sum(-1) - thumb_w.sum(-1) > 0.4, thumb_con)

        groups = torch.stack([(contacts_w[:, g] > 0.0).any(dim=-1) for g in NON_THUMB_FINGER_GROUPS], dim=-1)
        grasp = firm & (groups.float().sum(-1) >= 2) & thumb_con
        return touch, grasp

    def _force_signal_rewards(self, receiver_idx: int, holder_idx: int, arm_contact_matrix: torch.Tensor):
        '''
        The force-based reward signals (the project's own contribution, from
        the feature/force-reward-signals work) carried over unchanged from
        _phase_reward into the staged task. Term-for-term identical formulas;
        only the gating condition is re-expressed, since `new_receiver_grasped`
        no longer exists -- its equivalent is `stage >= 3` (receiver has a
        sustained multi-finger grasp).

          signal_1  grasp firmness   -- summed normalized contact force
          signal_2  thumb/finger force balance
          signal_instability  contact-pattern churn x force drop-off
          force_excess  safety penalty above F_safe

        force_excess is applied to BOTH hands here, where the baseline applied
        it only to the receiver. Under the staged task the holder is gripping
        for most of the episode, so a receiver-only penalty would let the
        holder crush the object for free.
        '''
        cfg = self.cfg
        stage = self.stage
        grasping = (stage >= 3).float()
        reward = torch.zeros(self.num_envs, device=self.device)
        log: dict = {}

        sl_r = self._gen3_contact_slice if receiver_idx == cfg.GEN3 else self._ur5e_contact_slice
        contacts_local = self.contacts[:, sl_r]
        force_local = self.force_norms[:, sl_r]
        force_w = torch.tanh(force_local / cfg.F_ref) * arm_contact_matrix

        if cfg.use_signal_1:
            # Only paid once the receiver is actually in contact, otherwise it
            # is identically zero anyway.
            signal_1_value = force_w[:, :-1].sum(-1)
            reward = reward + signal_1_value
            log["signal_1_firmness"] = signal_1_value

        if cfg.use_signal_2:
            thumb_force = force_w[:, THUMB_SLICE].sum(-1)
            finger_force = force_w[:, 0 : NUM_FINGER - 5].sum(-1) + cfg.palm_weight * force_w[:, NUM_FINGER : NUM_FINGER + 1].sum(-1)
            balance_imbalance = torch.abs(thumb_force - finger_force) / (thumb_force + finger_force + 1e-6)
            total_grasp_force = force_w[:, 0:NUM_FINGER].sum(-1)
            balance_active = (total_grasp_force > cfg.F_threshold).float()
            r_balance = -cfg.lambda_balance * balance_imbalance * grasping * balance_active
            reward = reward + r_balance
            log["signal_2_balance"] = r_balance

        if cfg.use_signal_instability:
            pattern_change = (contacts_local[:, :-1].float() - self.prev_contacts[:, sl_r][:, :-1]).abs().sum(-1)
            current_total = force_local[:, :-1].sum(-1)
            prev_total = self.prev_force_norms[:, sl_r][:, :-1].sum(-1)
            force_drop = torch.relu(prev_total - current_total)
            r_instability = -cfg.lambda_instability * pattern_change * force_drop * grasping
            reward = reward + r_instability
            log["signal_instability"] = r_instability

        if cfg.use_force_rewards:
            r_excess = torch.zeros(self.num_envs, device=self.device)
            for idx in (receiver_idx, holder_idx):
                sl = self._gen3_contact_slice if idx == cfg.GEN3 else self._ur5e_contact_slice
                r_excess = r_excess - cfg.lambda_force_excess * torch.relu(self.force_norms[:, sl][:, :-1] - cfg.F_safe).sum(-1)
            reward = reward + r_excess
            log["force_excess"] = r_excess

        return reward, log

    def _get_rewards(self) -> torch.Tensor:
        '''
        ALOHA-style staged single transfer (2026-08-15), replacing the
        continuous two-phase round-trip reward.

        Modelled on lerobot/act_aloha_sim_transfer_cube_human's task, which the
        user asked to take the setup from. That benchmark scores a 0-4 sparse
        ladder (touch / lift / other-gripper-touch / transferred) and calls
        success a SINGLE transfer, with the two arms acting in sequence rather
        than simultaneously. Ours:

          stage 0  holder (UR5e) brings the object to the handoff point
          stage 1  presented and held steady -> receiver (GEN3) is released to move
          stage 2  receiver's hand is touching the object
          stage 3  receiver has a sustained multi-finger grasp; holder opens
          stage 4  receiver still holds it after the holder let go  == SUCCESS

        The ladder is a one-way ratchet, so an env can never lose progress it
        has already banked. Between rungs the only dense signal is
        potential-based (prev_dist - dist) on whichever arm is currently
        active. That replaces Run A's `mod` gate, which paid approach reward
        ONLY on strictly-improving steps and is the leading suspect for the
        0.317m plateau: once the receiver stopped improving, its reward went to
        zero everywhere nearby and the gradient vanished. Potential-based
        shaping is dense, is symmetric (backing off costs exactly what
        approaching pays), and is the standard policy-invariant form.

        Also gone: the old `reward + contacts_w.sum(-1)` free-contact term,
        which paid for merely touching anything and accounted for most of Run
        A's flat ~79 ep_rew_mean without any task progress behind it.
        '''
        cfg = self.cfg
        # NOTE: prev_force_norms/prev_contacts are refreshed at the END of this
        # method, not the start. Diagnosed 2026-08-15: the baseline does the
        # clone on the first two lines of its _get_rewards
        # (bimanual_direct_env.py:580) and then reads self.prev_contacts at
        # line 705 to compute `pattern_change` -- by which point prev has
        # already been overwritten with the current values, so pattern_change
        # and force_drop are IDENTICALLY ZERO and signal_instability has never
        # contributed anything in any run, in this env or the baseline.
        arm_contact_matrix = self.cfg.contact_matrix[self._ur5e_contact_slice]  # identical to the gen3 segment by construction
        holder_idx, receiver_idx = cfg.UR5e, cfg.GEN3

        obj = self.scene.rigid_objects["object"].data.body_state_w[:, 0, :3]
        tips_receiver = self.scene.articulations[cfg.keys[receiver_idx]].data.body_state_w[
            :, self.finger_tips[receiver_idx], 0:7
        ].mean(1)[:, :3]

        # Both in metres, Euclidean. Deliberately NOT dual_quaternion_error:
        # index 0 of that blends position and rotation and was minimisable by
        # spinning the wrist (diagnosed 2026-08-15, see _phase_reward).
        present_dist = torch.norm(obj - self.handoff_pos, dim=-1)
        reach_dist = torch.norm(tips_receiver - obj, dim=-1)

        touch_receiver, grasp_receiver = self._grasp_flags(receiver_idx, arm_contact_matrix)
        stage = self.stage

        # ---- stage transitions (one-way ratchet) -------------------------
        self.present_streak = torch.where(
            present_dist < cfg.handoff_thresh, self.present_streak + 1, torch.zeros_like(self.present_streak)
        )
        self.receiver_grasp_streak = torch.where(
            grasp_receiver, self.receiver_grasp_streak + 1, torch.zeros_like(self.receiver_grasp_streak)
        )
        # Only counts once the holder has actually let go (stage >= 3), which
        # is what makes stage 4 a real transfer rather than a double grip.
        self.release_streak = torch.where(
            torch.logical_and(stage >= 3, grasp_receiver), self.release_streak + 1, torch.zeros_like(self.release_streak)
        )

        to1 = torch.logical_and(stage == 0, self.present_streak >= cfg.present_sustain_steps)
        to2 = torch.logical_and(stage == 1, touch_receiver)
        to3 = torch.logical_and(stage == 2, self.receiver_grasp_streak >= cfg.grasp_sustain_steps)
        to4 = torch.logical_and(stage == 3, self.release_streak >= cfg.release_sustain_steps)

        new_stage = stage.clone()
        for flag, value in ((to1, 1), (to2, 2), (to3, 3), (to4, 4)):
            new_stage = torch.where(flag, torch.full_like(stage, value), new_stage)
        self.stage = new_stage

        # ---- dense potential-based shaping, active arm only --------------
        # inf on the first step of an episode means "no baseline yet" -- pay 0
        # rather than treating inf-finite as an infinite improvement.
        holder_active = stage == 0
        receiver_active = torch.logical_and(stage >= 1, stage <= 2)
        d_present = torch.where(
            torch.isfinite(self.prev_present_dist), self.prev_present_dist - present_dist, torch.zeros_like(present_dist)
        ).clamp(-cfg.shaping_clip, cfg.shaping_clip)
        d_reach = torch.where(
            torch.isfinite(self.prev_reach_dist), self.prev_reach_dist - reach_dist, torch.zeros_like(reach_dist)
        ).clamp(-cfg.shaping_clip, cfg.shaping_clip)

        reward = cfg.w_present * d_present * holder_active.float() + cfg.w_reach * d_reach * receiver_active.float()

        self.prev_present_dist = present_dist
        self.prev_reach_dist = reach_dist

        # ---- sparse ladder bonuses, paid once on entry -------------------
        reward = (
            reward
            + cfg.bonus_present * to1.float()
            + cfg.bonus_touch * to2.float()
            + cfg.bonus_grasp * to3.float()
            + cfg.bonus_transfer * to4.float()
        )

        # ---- penalties ---------------------------------------------------
        # Holder must present a stable target while the receiver closes in.
        # Defaults to lambda 0 now that stage_gate_arms freezes the holder
        # outright -- runs 3908542/3908543 still measured 0.04-0.08/step here
        # WITH the freeze active, because zeroing the Cartesian increment does
        # not zero the arm's residual settling velocity. That was a constant
        # unavoidable drain the policy could only escape by ending the episode.
        holder_speed = torch.norm(
            self.scene.articulations[cfg.keys[holder_idx]].data.body_state_w[:, self.ee_jacobi_idx[holder_idx] + 1, 7:], dim=-1
        )
        holder_still_penalty = cfg.lambda_holder_still * holder_speed * receiver_active.float()
        reward = reward - holder_still_penalty

        # Truncation penalty. Diagnosed 2026-08-15 from runs 3908542/3908543:
        # ep_len_mean collapsed from 399 to 1-30 steps in both, while stage2
        # and reward peaked around iteration 35-98 and then decayed. Per-step
        # reward was net negative (~-0.06 to -0.10, dominated by the holder
        # penalty above), so ending the episode immediately scored ~0 and was
        # strictly better than playing it out -- the policy was deliberately
        # driving an arm into the ground/height limit to stop the bleeding.
        # Charging for a truncated ending makes surviving the better option
        # again; combined with lambda_holder_still=0 the per-step floor is now
        # ~0 rather than negative.
        truncation_penalty = cfg.penalty_truncation * self._last_truncated.float()
        reward = reward - truncation_penalty

        arm_arm_contact = (self.contacts[:, -1] > 0.0).float()
        reward = reward - cfg.lambda_arm_collision * arm_arm_contact

        # ---- force reward signals (carried over from the baseline) --------
        force_reward, force_log = self._force_signal_rewards(receiver_idx, holder_idx, arm_contact_matrix)
        reward = reward + force_reward

        self.transfer_complete = self.stage >= 4
        # Kept in sync so inherited BimanualDirect code and the weld/release
        # helpers agree on who is holding the object.
        self.obj_reached = self.stage >= 3
        self.round_trip_complete = self.transfer_complete

        log = self.extras.setdefault("log", {})
        log["metrics/stage_mean"] = self.stage.float().mean().item()
        for k in range(1, 5):
            log[f"metrics/stage{k}_fraction"] = (self.stage >= k).float().mean().item()
        log["metrics/present_dist"] = present_dist.mean().item()
        log["metrics/hand_obj_dist"] = reach_dist.mean().item()
        log["metrics/arm_arm_collision"] = arm_arm_contact.mean().item()
        log["metrics/success_rate"] = self.transfer_complete.float().mean().item()
        log["reward_terms/shaping_present"] = (cfg.w_present * d_present * holder_active.float()).mean().item()
        log["reward_terms/shaping_reach"] = (cfg.w_reach * d_reach * receiver_active.float()).mean().item()
        log["reward_terms/holder_still_penalty"] = holder_still_penalty.mean().item()
        log["reward_terms/truncation_penalty"] = truncation_penalty.mean().item()
        for key, value in force_log.items():
            log[f"reward_terms/{key}"] = value.mean().item() if torch.is_tensor(value) else value

        # Refresh AFTER every read of the previous step's values -- see the
        # ordering note at the top of this method.
        self.prev_force_norms = self.force_norms.clone()
        self.prev_contacts = self.contacts.clone().float()

        return reward

    def _get_rewards_round_trip_legacy(self) -> torch.Tensor:
        '''
        The pre-2026-08-15 two-phase round-trip reward, kept for reference and
        for the eventual curriculum step back up to a full round trip. Not
        called. success_rate was 0 in every run this ever drove.
        '''
        self.prev_force_norms = self.force_norms.clone()
        self.prev_contacts = self.contacts.clone().float()

        arm_contact_matrix = self.cfg.contact_matrix[self._ur5e_contact_slice]  # identical to the gen3 segment by construction

        phase0 = self.phase == 0
        phase1 = self.phase == 1

        reward0, hand_dist0, target_dist0, gen3_grasped_new, just_trans0, reached_target0, log0, gen3_streak_new = self._phase_reward(
            receiver_idx=self.cfg.GEN3, holder_idx=self.cfg.UR5e,
            contacts_local=self.contacts[:, self._gen3_contact_slice],
            force_norms_local=self.force_norms[:, self._gen3_contact_slice],
            prev_contacts_local=self.prev_contacts[:, self._gen3_contact_slice],
            prev_force_norms_local=self.prev_force_norms[:, self._gen3_contact_slice],
            arm_contact_matrix=arm_contact_matrix,
            target_pose=self.cfg.target_pose, prev_dist=self.prev_dist, prev_dist_target=self.prev_dist_target,
            receiver_grasped=self.obj_reached,
            contact_streak=self.gen3_contact_streak,
        )
        reward1, hand_dist1, target_dist1, ur5e_grasped_new, just_trans1, reached_target1, log1, ur5e_streak_new = self._phase_reward(
            receiver_idx=self.cfg.UR5e, holder_idx=self.cfg.GEN3,
            contacts_local=self.contacts[:, self._ur5e_contact_slice],
            force_norms_local=self.force_norms[:, self._ur5e_contact_slice],
            prev_contacts_local=self.prev_contacts[:, self._ur5e_contact_slice],
            prev_force_norms_local=self.prev_force_norms[:, self._ur5e_contact_slice],
            arm_contact_matrix=arm_contact_matrix,
            target_pose=self.cfg.target_pose_ur5e, prev_dist=self.prev_dist_ur5e, prev_dist_target=self.prev_dist_target_ur5e,
            receiver_grasped=self.ur5e_grasped,
            contact_streak=self.ur5e_contact_streak,
        )

        self.obj_reached = torch.where(phase0, gen3_grasped_new, self.obj_reached)
        self.ur5e_grasped = torch.where(phase1, ur5e_grasped_new, self.ur5e_grasped)
        self.gen3_contact_streak = torch.where(phase0, gen3_streak_new, self.gen3_contact_streak)
        self.ur5e_contact_streak = torch.where(phase1, ur5e_streak_new, self.ur5e_contact_streak)
        self.prev_dist = torch.where(phase0, hand_dist0, self.prev_dist)
        self.prev_dist_target = torch.where(phase0, target_dist0, self.prev_dist_target)
        self.prev_dist_ur5e = torch.where(phase1, hand_dist1, self.prev_dist_ur5e)
        self.prev_dist_target_ur5e = torch.where(phase1, target_dist1, self.prev_dist_target_ur5e)

        # Team reward: only the active phase's contribution counts per env
        # (envs in the batch can be in different phases at once).
        reward = torch.where(phase0, reward0, torch.zeros_like(reward0)) + torch.where(phase1, reward1, torch.zeros_like(reward1))

        # Phase transition: GEN3 has firmly grasped and delivered close enough
        # to target_pose -> hand off to phase 1. Team bonus for a clean
        # transfer, on top of each phase's own grasp bonus.
        advance = torch.logical_and(phase0, torch.logical_and(gen3_grasped_new, reached_target0))
        self.phase = torch.where(advance, torch.ones_like(self.phase), self.phase)
        reward = reward + self.cfg.bonus_obj_reach * advance.float()

        # Arm-to-arm collision penalty. self.contacts' final column is the
        # robot1_w_robot2 sensor (see update_collisions_bidirectional). It had
        # weight 0 in contact_matrix and was absent from _get_dones, so the
        # policy had no incentive whatsoever to avoid driving the two arms
        # through each other -- which the 0.60m separation actively encouraged.
        arm_arm_contact = (self.contacts[:, -1] > 0.0).float()
        reward = reward - self.cfg.lambda_arm_collision * arm_arm_contact
        self.extras.setdefault("log", {})["metrics/arm_arm_collision"] = arm_arm_contact.mean().item()

        self.round_trip_complete = torch.logical_and(phase1, torch.logical_and(ur5e_grasped_new, reached_target1))

        self._log_phase(log0, "")
        self._log_phase(log1, "_ur5e")
        self.extras["log"]["metrics/success_rate"] = self.round_trip_complete.float().mean().item()
        self.extras["log"]["metrics/phase1_fraction"] = phase1.float().mean().item()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        out_of_bounds_ur5e = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.UR5e] + 1, 7:], dim=-1) > self.cfg.velocity_limit
        out_of_bounds_gen3 = torch.norm(self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.GEN3] + 1, 7:], dim=-1) > self.cfg.velocity_limit
        out_of_bounds = torch.logical_or(out_of_bounds_ur5e, out_of_bounds_gen3)

        ur5e_falling = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.UR5e] + 1, 2] < self.cfg.gen3_height_limit
        gen3_falling = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[:, self.ee_jacobi_idx[self.cfg.GEN3] + 1, 2] < self.cfg.gen3_height_limit
        object_falling = self.scene.rigid_objects["object"].data.body_state_w[:, 0, 2] < self.cfg.object_height_limit
        falling = torch.logical_or(torch.logical_or(ur5e_falling, gen3_falling), object_falling)

        # Column order fixed by update_collisions_bidirectional(): ur5e_w_ground, gen3_w_ground first.
        ground_contact = torch.logical_or(self.contacts[:, 0] > 0.0, self.contacts[:, 1] > 0.0)

        truncated = torch.logical_or(torch.logical_or(falling, out_of_bounds), ground_contact)
        # Success is now a SINGLE completed transfer (stage 4), matching the
        # ALOHA transfer-cube benchmark's own success definition, rather than
        # the round trip -- which was strictly harder than the published task
        # and never fired once in any run.
        terminated = torch.logical_or(time_out, self.transfer_complete)

        success = self.transfer_complete & ~truncated
        indet = self.transfer_complete & truncated
        fail = ~self.transfer_complete & (terminated | truncated)

        # Stashed for _get_rewards, which runs immediately after _get_dones in
        # DirectRLEnv.step and charges cfg.penalty_truncation for ending an
        # episode this way. See the reward's own comment for why.
        self._last_truncated = truncated

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["metrics/outcome_success"] = success.float().mean().item()
        self.extras["log"]["metrics/outcome_indetermination"] = indet.float().mean().item()
        self.extras["log"]["metrics/outcome_fail"] = fail.float().mean().item()
        self.extras["log"]["metrics/out_of_bounds"] = out_of_bounds.float().mean().item()
        # Per-cause truncation logging (2026-08-15): runs 3908542/3908543 saw
        # ep_len_mean collapse from 399 to 1-30 steps while out_of_bounds
        # stayed at exactly 0, so the cause was falling or ground contact --
        # but the three were only ever logged as one aggregate, leaving no way
        # to tell which. Split out so the next run answers it directly.
        self.extras["log"]["metrics/trunc_ground_contact"] = ground_contact.float().mean().item()
        self.extras["log"]["metrics/trunc_arm_falling"] = torch.logical_or(ur5e_falling, gen3_falling).float().mean().item()
        self.extras["log"]["metrics/trunc_object_falling"] = object_falling.float().mean().item()
        self.extras["log"]["metrics/trunc_any"] = truncated.float().mean().item()

        self._update_curriculum(success, torch.logical_or(truncated, terminated))
        self.extras["log"]["metrics/curriculum_level"] = float(self.curriculum_level)
        self.extras["log"]["metrics/curriculum_start_frac"] = float(
            self.cfg.curriculum_start_frac[min(self.curriculum_level, len(self.cfg.curriculum_start_frac) - 1)]
        )

        return truncated, terminated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Reuses BimanualDirect._reset_idx wholesale: resets both arms to
        # their (randomized, for UR5e) starting poses, resets the object
        # relative to UR5e's end effector, resets the IK controller, and
        # zeroes the doubled contact/force tensors (already correctly sized
        # since self.num_contacts is computed generically from
        # len(cfg.contact_sensors_dict) in BimanualDirect.__init__).
        super()._reset_idx(env_ids)

        self.phase[env_ids] = 0
        self.ur5e_grasped[env_ids] = False
        self.gen3_contact_streak[env_ids] = 0
        self.ur5e_contact_streak[env_ids] = 0
        self.round_trip_complete[env_ids] = False
        self.prev_dist_ur5e[env_ids] = torch.inf
        self.prev_dist_target_ur5e[env_ids] = torch.inf

        # --- staged single transfer ---------------------------------------
        self.stage[env_ids] = 0
        self.present_streak[env_ids] = 0
        self.receiver_grasp_streak[env_ids] = 0
        self.release_streak[env_ids] = 0
        self.transfer_complete[env_ids] = False
        self.prev_present_dist[env_ids] = torch.inf
        self.prev_reach_dist[env_ids] = torch.inf

        # Handoff point = midpoint between the two end effectors at reset,
        # raised by cfg.handoff_height_offset. Computed per episode rather
        # than hardcoded because UR5e's start pose is randomized, so a fixed
        # world coordinate would sometimes be trivially close to the holder
        # and sometimes outside its reach. Both arms are identical GEN3s at
        # a fixed separation, so the midpoint is reachable by construction.
        ee_holder = self.scene.articulations[self.cfg.keys[self.cfg.UR5e]].data.body_state_w[
            env_ids, self.ee_jacobi_idx[self.cfg.UR5e] + 1, 0:3
        ]
        ee_receiver = self.scene.articulations[self.cfg.keys[self.cfg.GEN3]].data.body_state_w[
            env_ids, self.ee_jacobi_idx[self.cfg.GEN3] + 1, 0:3
        ]
        handoff = 0.5 * (ee_holder + ee_receiver)
        handoff[:, 2] = handoff[:, 2] + self.cfg.handoff_height_offset
        self.handoff_pos[env_ids] = handoff

        self._apply_curriculum_reset(env_ids)

    def _apply_curriculum_reset(self, env_ids: Sequence[int]):
        '''
        Reverse curriculum: move the receiver's starting pose a fraction of the
        way toward the object's grasp point, and start the episode already at
        stage 1 (object presented, giver frozen, receiver free to move).

        At the easiest level the receiver begins essentially on the object, so
        the ONLY thing left to learn is closing the hand and holding -- the
        exact transition that jobs 3908719/3908720 failed at (66% reached
        "touching", 0-3% reached "gripped"). As the pooled success rate passes
        cfg.curriculum_promote_rate the start pose walks back toward the normal
        reset, and the final level (frac 0.0) IS the unmodified task, so a run
        that reaches the last level is directly comparable to those two.

        Only the receiver is moved. Moving the meeting point instead would be
        simpler but cannot work: the two arms sit ~0.37m apart, so any meeting
        point that is easy for the receiver to reach is correspondingly far for
        the giver, and the curriculum would make one arm's job easier only by
        making the other's harder.

        LIMITATION (revisit after the first videos): only the POSITION is
        interpolated -- the wrist keeps its sampled home orientation. There is
        no defined "correct" grasp orientation in the config to interpolate
        toward. If the videos show the hand arriving at the object badly
        oriented at the easy levels, that is this, and it needs a target
        orientation rather than more training.
        '''
        cfg = self.cfg
        if not cfg.use_reverse_curriculum:
            return
        level = min(self.curriculum_level, len(cfg.curriculum_start_frac) - 1)
        frac = cfg.curriculum_start_frac[level]
        if frac <= 0.0:
            return  # hardest level == the normal reset, leave it untouched

        receiver = cfg.GEN3
        key = cfg.keys[receiver]

        # Object's world pose, derived ANALYTICALLY rather than read back from
        # the sim. super()._reset_idx() has just written both the arm joint
        # states and the object root pose this same call, and those writes are
        # not guaranteed to be visible in data.body_state_w until the next sim
        # step -- reading it here can return the PREVIOUS episode's pose and
        # would aim the curriculum placement at a stale target. The chain
        # root -> commanded home EE -> fixed object offset uses only the
        # articulation root (which never moves) plus values we just set.
        giver = cfg.UR5e
        giver_root_w = self.scene.articulations[cfg.keys[giver]].data.root_state_w[:, 0:7]
        home_giver = self.reset_robot_poses_r[giver]
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            t01=giver_root_w[:, :3], q01=giver_root_w[:, 3:],
            t12=home_giver[:, :3], q12=home_giver[:, 3:],
        )
        obj_pos_w, obj_quat_w = combine_frame_transforms(
            t01=ee_pos_w, q01=ee_quat_w,
            t12=cfg.obj_pos_trans, q12=cfg.obj_quat_trans,
        )
        obj_pose_w = torch.cat((obj_pos_w, obj_quat_w), dim=-1)

        # Object's grasp point, expressed in the receiver's root frame.
        gp_pos_w, gp_quat_w = combine_frame_transforms(
            t01=obj_pose_w[:, :3], q01=obj_pose_w[:, 3:],
            t12=cfg.grasp_obs_obj_pos_trans, q12=cfg.grasp_obs_obj_quat_trans,
        )
        root_w = self.scene.articulations[key].data.root_state_w[:, 0:7]
        gp_pos_r, _ = subtract_frame_transforms(
            t01=root_w[:, :3], q01=root_w[:, 3:], t02=gp_pos_w, q02=gp_quat_w
        )

        # Interpolate home -> grasp point. Orientation is left at home (see docstring).
        home = self.reset_robot_poses_r[receiver]
        target = home.clone()
        target[:, :3] = home[:, :3] + frac * (gp_pos_r - home[:, :3])

        # Same IK-and-write pattern the parent uses for the normal reset pose.
        self.controller.set_command(target)
        ee_pos_r, ee_quat_r, jacobian, joint_pos = self._get_ee_pose(receiver)
        arm_q = self.controller.compute(ee_pos_r, ee_quat_r, jacobian, joint_pos)
        joint_pos_full = torch.cat(
            (arm_q, self.default_joint_pos[receiver][:, len(cfg.joints[receiver]):]), dim=-1
        )
        joint_vel = self.scene.articulations[key].data.default_joint_vel
        self.scene.articulations[key].write_joint_state_to_sim(
            joint_pos_full[env_ids], joint_vel[env_ids], None, env_ids
        )
        # Keep the running IK/action target consistent with where the arm now
        # actually is, otherwise the first perform_increment would command it
        # straight back to the un-curriculumed home pose.
        self.reset_joint_positions[receiver][env_ids] = joint_pos_full[env_ids]
        self.actions[receiver][env_ids] = joint_pos_full[env_ids]

        # Begin already presented: _gate_arm_motion freezes the giver at
        # stage >= 1 and releases the receiver, which is the state the easy
        # levels are meant to drill.
        self.stage[env_ids] = 1
        self.handoff_pos[env_ids] = obj_pose_w[env_ids, :3]

    def _update_curriculum(self, success: torch.Tensor, done: torch.Tensor):
        '''
        Pooled promote/demote on a rolling window of completed episodes.
        Demotion is deliberate: every previous run's failure mode was sitting
        at 0% for hundreds of iterations with no way back, so a level that
        turns out to be too hard should drop rather than stall.
        '''
        cfg = self.cfg
        if not cfg.use_reverse_curriculum:
            return
        n_done = int(done.sum().item())
        if n_done == 0:
            return
        self._curr_ep_count += n_done
        self._curr_succ_count += int((success & done).sum().item())
        if self._curr_ep_count < cfg.curriculum_window:
            return

        rate = self._curr_succ_count / max(self._curr_ep_count, 1)
        max_level = len(cfg.curriculum_start_frac) - 1
        if rate >= cfg.curriculum_promote_rate and self.curriculum_level < max_level:
            self.curriculum_level += 1
        elif rate < cfg.curriculum_demote_rate and self.curriculum_level > 0:
            self.curriculum_level -= 1
        self._curr_ep_count = 0
        self._curr_succ_count = 0
