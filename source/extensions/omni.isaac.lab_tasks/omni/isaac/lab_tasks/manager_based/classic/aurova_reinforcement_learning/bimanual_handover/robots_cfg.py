"""Configuration for the UR5e robot"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from math import pi
## 
# Configuration
##

GEN3_4f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/config/usd/gen3_4f.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_joint_1": 0.0,
            "arm_joint_2": -pi/8,
            "arm_joint_3": 0.0,
            "arm_joint_4": 3*pi/4,
            "arm_joint_5": 0.0,
            "arm_joint_6": -pi/6,
            "arm_joint_7": 5*pi/4 - pi,
            "joint_12_0": 0.263,
        },
        pos = (-1.25, 0.0, 0.0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr = ["arm_.*"],
            velocity_limit = 100.0,
            effort_limit = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit = 100.0,
            effort_limit = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)


# UFactory xArm7 carrying the same 16-DOF 4-finger hand -- the lab's hardware. Built from
# config/urdf/xarm7_4f.urdf (provenance: config/urdf/xarm_src/README.md): GEN3's hand,
# contact links and mount kept unchanged, arm replaced. Converted to USD with the same
# settings as gen3_4f.usd. The arm actuators are GEN3_4f_CFG's: robot bodies have gravity
# disabled in this task, so the PD gains only track commanded motion, and keeping them
# identical keeps the controller behaviour of the GEN3 runs that worked.
#
# The default arm posture puts hand_palm_link at exactly the pose GEN3's validated handover
# posture did (identity orientation at (0.14, 0, 0.85) in the arm base frame: palm front +X
# at the other arm, palms 0.32 m apart). Solved with URDF forward kinematics + damped least
# squares over many seeds, UFactory joint limits respected; chosen for the largest worst-case
# dexterity (Jacobian min singular value 0.208 vs 0.143 for GEN3's own pose) with >= 0.32 m
# between the two arms' links. Must still be validated in simulation (grip, palm facing,
# no arm contact) -- see diag/validate_xarm_pose.py.
XARM7_HANDOVER_Q = [-0.6799084361, -1.148776091, 0.2116508428, 0.8330124805,
                    -2.9324690166, 1.1761795864, 1.2938908653]
XARM7_4f_CFG = GEN3_4f_CFG.replace(
    spawn=GEN3_4f_CFG.spawn.replace(
        usd_path=GEN3_4f_CFG.spawn.usd_path.replace("gen3_4f.usd", "xarm7_4f.usd"),
    ),
    init_state=GEN3_4f_CFG.init_state.replace(
        joint_pos={**{f"arm_joint_{k + 1}": v for k, v in enumerate(XARM7_HANDOVER_Q)}, "joint_12_0": 0.263},
    ),
)

UR5e_4f_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/config/usd/ur5e_4f_ros2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_shoulder_pan_joint": 0.0,
            "arm_shoulder_lift_joint": -pi/2,
            "arm_elbow_joint": -pi/2,
            "arm_wrist_1_joint": 0.0,
            "arm_wrist_2_joint": pi/2,
            "arm_wrist_3_joint": pi/4,
            "joint_0_0": 0.0,
            "joint_1_0": 0.263*6,
            "joint_2_0": 0.263*5,
            "joint_3_0": 0.263*2.3,
            "joint_4_0": 0.0,
            "joint_5_0": 0.263*6,
            "joint_6_0": 0.263*5,
            "joint_7_0": 0.263*2.3,
            "joint_8_0": 0.0,
            "joint_9_0": 0.263*6,
            "joint_10_0": 0.263*5,
            "joint_11_0": 0.263*2.3,
            "joint_12_0": 0.263, # zero position is 0.263
            "joint_13_0": 0.0,
            "joint_14_0": 0.263*5,
            "joint_15_0": 0.263*5,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr = ["arm_.*"],
            velocity_limit = 100.0,
            effort_limit = 87.0,
            stiffness = 800.0,
            damping = 40.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_0"],
            velocity_limit = 100.0,
            effort_limit = 0.5,
            stiffness = 3.0,
            damping = 0.1,
            friction = 0.01,
        ), 
    },

)

