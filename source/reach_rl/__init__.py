from gymnasium.envs.registration import register

register(
     id="reach_rl/ReachRL-v0",
     entry_point="reach_rl.envs:ReachRL")
