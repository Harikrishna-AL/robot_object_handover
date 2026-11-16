# xArm7 + 4-finger hand model

`../xarm7_4f.urdf` = the hand subtree of `../gen3_4f.urdf` (hand, contact links and the
`palm_2_ur` mount, all unchanged) on UFactory's xArm7 instead of the GEN3 arm.

Regenerate:
1. `git clone https://github.com/xArm-Developer/xarm_ros2` (used: commit ac736c9, 2026-10-20)
2. In `xarm7_only.urdf.xacro` replace `XARM_DESCRIPTION` with the path to `xarm_ros2/xarm_description`,
   and resolve `$(find xarm_description)` in its xacro files to the same path.
3. `pip install xacro && xacro xarm7_only.urdf.xacro > xarm7_only.urdf`   (model_num=0, standard meshes)
4. `python ../build_xarm7_4f.py ../gen3_4f.urdf xarm7_only.urdf ../xarm7_4f.urdf`

xArm joints are renamed `joint1..7 -> arm_joint_1..7`. Meshes live in `../../meshes/xarm/`
(BSD-3, see `LICENSE_xarm_ros2`).

**Unverified against the real robot:** the hand mount (`palm_2_ur`: 0.1175 m out along the flange
axis, yaw 5.5 rad) is the GEN3 adapter's offset. Measure the lab's real xArm adapter and update it.
If the lab's xArm7 is a newer revision (model number >= 1305), regenerate with that `model_num`.
