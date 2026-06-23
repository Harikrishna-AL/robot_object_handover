# Object Handover

We plan to explore possible improvements in the task of hading over or taking an object from a target coordinate. 

## Installation

```
cd docker
```

To start the containers:

```shell 
python3 container.py start
```

To enter the containers:

```shell
python3 container.py enter
```

## To Train

```shell
./isaaclab.sh -p source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/train.py --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1024 --headless --experience /workspace/isaaclab/_isaac_sim/apps/omni.isaac.sim.python.kit
```

## To eval (with live stream)

```shell
./isaaclab.sh -p source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train/eval.py --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1  --experience /workspace/isaaclab/_isaac_sim/apps/omni.isaac.sim.python.kit --enable_cameras --model_dir sb3/Isaac-Bimanual-Direct-reach-v0/2026-06-17_10-03-49 --headless --livestream 2
```