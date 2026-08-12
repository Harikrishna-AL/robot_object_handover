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
  + role one-hot [is_holder, is_receiver] (2)
= 26 (BimanualBidirectionalCfg.arm_obs_dim). The action is
[UR5e action (9) | GEN3 action (9)], same per-arm meaning as BimanualDirect's
single 9-dim action. If either layout changes, update
BimanualBidirectionalCfg.arm_obs_dim/role_obs_dim and
sb3_ppo_cfg_bidirectional.yaml's policy_kwargs together.
'''

NUM_FINGER = len(FINGER_LINKS)
THUMB_SLICE = slice(NUM_FINGER - 5, NUM_FINGER)  # finger_4 (2 links) + thumb (3 links), mirrors BimanualDirect's `-8:-3` slice into its 18-wide global array
NUM_GLOBAL_CONTACTS = 2  # ur5e_w_ground, gen3_w_ground

REWARD_TERM_KEYS = {
    "approach", "target", "bonus", "signal_1_firmness", "signal_2_balance",
    "signal_instability", "velocity_penalty", "force_excess",
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
        self.prev_dist_ur5e = torch.full((self.num_envs,), torch.inf, device=self.device)
        self.prev_dist_target_ur5e = torch.full((self.num_envs,), torch.inf, device=self.device)

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

        self.perform_increment(idx=self.cfg.UR5e, actions=actions_ur5e)
        self.perform_increment(idx=self.cfg.GEN3, actions=actions_gen3)

        self._apply_releases()

    def _apply_releases(self):
        '''
        Mirrors BimanualDirect's "holder opens its hand once the receiver
        grasps" behaviour (bimanual_direct_env.py:258) in both directions:
        UR5e opens once GEN3 grasps (phase 0, same as the original task),
        GEN3 opens once UR5e grasps (phase 1). Hand joints are always the
        last 16 columns of self.actions[idx] regardless of arm DOF (6 for
        UR5e, 7 for GEN3), so -16: is used rather than a fixed offset.
        '''
        phase0_release = torch.logical_and(self.phase == 0, self.obj_reached)
        self.actions[self.cfg.UR5e][phase0_release, -16:] = self.open_hand_joints

        phase1_release = torch.logical_and(self.phase == 1, self.ur5e_grasped)
        self.actions[self.cfg.GEN3][phase1_release, -16:] = self.open_hand_joints

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

        role_ur5e = torch.stack(((self.phase == 0).float(), (self.phase == 1).float()), dim=-1)
        role_gen3 = torch.stack(((self.phase == 1).float(), (self.phase == 0).float()), dim=-1)

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
        just_transitioned, reached_target, log).
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

        just_transitioned = torch.logical_and(contacts_flag, torch.logical_not(receiver_grasped))
        bonus = receiver_grasped.clone()
        new_receiver_grasped = torch.logical_or(contacts_flag, receiver_grasped)
        bonus = torch.logical_and(new_receiver_grasped.clone(), torch.logical_not(bonus))

        # Generalized "object arrived at target" check: original task used an
        # ad-hoc `obj_pose[:, 0] < 0.05` (a heuristic tuned to GEN3's specific
        # approach frame -- see bimanual_direct_env.py:649, whose own comment
        # shows this dual-quaternion-distance form as the considered
        # alternative). That heuristic doesn't transfer to UR5e's differently
        # oriented approach, so both phases use this symmetric form instead.
        reached_target = obj_target_dist[:, 1] < cfg.obj_reach_target_thresh

        dist = hand_obj_dist[:, 0] * torch.logical_not(new_receiver_grasped).int() + obj_target_dist[:, 0] * new_receiver_grasped.int()
        prev_dist_eff = prev_dist * torch.logical_not(new_receiver_grasped).int() + prev_dist_target * new_receiver_grasped.int()

        pre_mod = torch.logical_and(tips_dist > obj_dist, hand_obj_dist_back[:, 0] > hand_obj_dist[:, 0])
        mod = 2 * (torch.logical_and(dist < prev_dist_eff, pre_mod)) - 1

        rew_scale_hand_obj = cfg.rew_scale_hand_obj * torch.ones(self.num_envs, device=device)
        rew_scale_hand_obj = rew_scale_hand_obj / (contacts_local[:, :-1].sum(-1) + 1)

        reward_1 = mod * rew_scale_hand_obj * torch.exp(-2 * hand_obj_dist[:, 0]) / (1 + 2 * (torch.logical_not(pre_mod)).int())
        reward_2 = cfg.rew_scale_obj_target * torch.exp(-2 * obj_target_dist[:, 0])

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
            velocity_penalty = cfg.lambda_vel * (excess_vel**2) * torch.exp(-cfg.k_decay * hand_obj_dist[:, 0]) * torch.logical_not(new_receiver_grasped).float()
            reward = reward - velocity_penalty
            log["velocity_penalty"] = velocity_penalty

        if cfg.use_force_rewards:
            excess_force = torch.relu(force_norms_local[:, :-1] - cfg.F_safe)
            r_excess = -cfg.lambda_force_excess * excess_force.sum(-1)
            reward = reward + r_excess
            log["force_excess"] = r_excess

        log["hand_obj_dist"] = hand_obj_dist[:, 0]
        log["obj_target_dist"] = obj_target_dist[:, 0]
        log["ee_vel_norm"] = ee_vel_norm
        log["contact_forces"] = contacts_w.sum(-1)
        log["force_at_transition"] = (
            force_norms_local[:, :NUM_FINGER][just_transitioned].sum(-1).mean().item() if just_transitioned.any() else 0.0
        )
        log["velocity_at_transition"] = ee_vel_norm[just_transitioned].mean().item() if just_transitioned.any() else 0.0

        return reward, hand_obj_dist[:, 0], obj_target_dist[:, 0], new_receiver_grasped, just_transitioned, reached_target, log

    def _log_phase(self, log: dict, suffix: str):
        if "log" not in self.extras:
            self.extras["log"] = dict()
        for key, value in log.items():
            val = value.mean().item() if torch.is_tensor(value) else value
            prefix = "reward_terms" if key in REWARD_TERM_KEYS else "metrics"
            self.extras["log"][f"{prefix}/{key}{suffix}"] = val

    def _get_rewards(self) -> torch.Tensor:
        self.prev_force_norms = self.force_norms.clone()
        self.prev_contacts = self.contacts.clone().float()

        arm_contact_matrix = self.cfg.contact_matrix[self._ur5e_contact_slice]  # identical to the gen3 segment by construction

        phase0 = self.phase == 0
        phase1 = self.phase == 1

        reward0, hand_dist0, target_dist0, gen3_grasped_new, just_trans0, reached_target0, log0 = self._phase_reward(
            receiver_idx=self.cfg.GEN3, holder_idx=self.cfg.UR5e,
            contacts_local=self.contacts[:, self._gen3_contact_slice],
            force_norms_local=self.force_norms[:, self._gen3_contact_slice],
            prev_contacts_local=self.prev_contacts[:, self._gen3_contact_slice],
            prev_force_norms_local=self.prev_force_norms[:, self._gen3_contact_slice],
            arm_contact_matrix=arm_contact_matrix,
            target_pose=self.cfg.target_pose, prev_dist=self.prev_dist, prev_dist_target=self.prev_dist_target,
            receiver_grasped=self.obj_reached,
        )
        reward1, hand_dist1, target_dist1, ur5e_grasped_new, just_trans1, reached_target1, log1 = self._phase_reward(
            receiver_idx=self.cfg.UR5e, holder_idx=self.cfg.GEN3,
            contacts_local=self.contacts[:, self._ur5e_contact_slice],
            force_norms_local=self.force_norms[:, self._ur5e_contact_slice],
            prev_contacts_local=self.prev_contacts[:, self._ur5e_contact_slice],
            prev_force_norms_local=self.prev_force_norms[:, self._ur5e_contact_slice],
            arm_contact_matrix=arm_contact_matrix,
            target_pose=self.cfg.target_pose_ur5e, prev_dist=self.prev_dist_ur5e, prev_dist_target=self.prev_dist_target_ur5e,
            receiver_grasped=self.ur5e_grasped,
        )

        self.obj_reached = torch.where(phase0, gen3_grasped_new, self.obj_reached)
        self.ur5e_grasped = torch.where(phase1, ur5e_grasped_new, self.ur5e_grasped)
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
        terminated = torch.logical_or(time_out, self.round_trip_complete)

        success = self.round_trip_complete & ~truncated
        indet = self.round_trip_complete & truncated
        fail = ~self.round_trip_complete & (terminated | truncated)

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["metrics/outcome_success"] = success.float().mean().item()
        self.extras["log"]["metrics/outcome_indetermination"] = indet.float().mean().item()
        self.extras["log"]["metrics/outcome_fail"] = fail.float().mean().item()
        self.extras["log"]["metrics/out_of_bounds"] = out_of_bounds.float().mean().item()

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
        self.round_trip_complete[env_ids] = False
        self.prev_dist_ur5e[env_ids] = torch.inf
        self.prev_dist_target_ur5e[env_ids] = torch.inf
