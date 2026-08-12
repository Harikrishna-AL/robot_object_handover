from __future__ import annotations

import torch

from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils import configclass

from .bimanual_direct_env_cfg import BimanualDirectCfg

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


@configclass
class BimanualBidirectionalCfg(BimanualDirectCfg):

    # ---- Action / observation sizing ----
    # 9 actions/arm (3 translation + 3 rotation + 3 hand control -- identical
    # per-arm meaning to BimanualDirectCfg's single-arm action). Action vector
    # is [UR5e (9) | GEN3 (9)], fixed order.
    num_actions = 9 * 2

    # Per-arm observation block: own ee pose rel. own base (7) + object/grasp
    # point pose rel. own base (7) + own selected hand-joint state (3) + other
    # arm's ee pose rel. own base (7) + role one-hot [is_holder, is_receiver]
    # (2) = 26. Both blocks concatenated in fixed order [UR5e, GEN3] -> 52.
    # SharedArmActorCriticPolicy reads arm_obs_dim/role_obs_dim from
    # policy_kwargs in sb3_ppo_cfg_bidirectional.yaml -- these three numbers
    # must stay in sync with the block layout in
    # BimanualBidirectional._get_observations if that layout ever changes.
    arm_obs_dim = 26
    role_obs_dim = 2
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
    contact_sensors_dict["robot1_w_robot2"] = ContactSensorCfg(
        prim_path="/World/envs/env_.*/" + cfg.keys[cfg.GEN3] + "/.*_link",
        update_period=0.001,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            f"/World/envs/env_{i}/{cfg.keys[cfg.UR5e]}/{joint}"
            for i in range(num_envs)
            for joint in cfg.links[cfg.UR5e]
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
