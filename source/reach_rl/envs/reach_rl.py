import gymnasium as gym
import numpy as np
from math import pi
import random
import torch
from .py_dq.src.lie import *
from .py_dq.src.interpolators import *
from .py_dq.src.distances import *


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


 
# Gym environment
class ReachRL(gym.Env):
    # Environment Metadata
    metadata = {'render_modes': ['human', 'rgb_array'],
                'render_fps': 60,
                'show': False}  
  
    def __init__(self, render_mode="human", data = {}): 

        # Checks if the selected render mode is within the possibilities
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # --- Mapping configuration ---
        DQ = 0
        EULER = 1
        QUAT = 2
        MAT = 3
        AXIS = 4


        # Size of the Lie algebra
        sizes = [[8, 6, 7, 12, 7], [6]*5]
        
        representation = DQ
        mapping = 1
        self.size_lie = sizes[int(mapping != 0)][representation]
        self.size_group = sizes[0][representation]
        distance = 1

        scalings = [[[1, 1], [0.02, 0.002]],
                    [[1, 1], [None, None]],
                    [[1, 1], [None, None]],
                    [[1, 1], [None, None]],
                    [[1, 1], [None, None]]]

        self.action_scaling = scalings[representation][mapping]
        




        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.action_space = gym.spaces.box.Box(low  = -10,
                                               high = 10, 
                                               shape = (self.size_lie, ),
                                               dtype=np.float32)
            
    
        self.observation_space = gym.spaces.box.Box(low  = -10,
                                                    high = 10,
                                                    shape = (self.size_lie,) ,
                                                    dtype=np.float32)
        
        


        self.ee_init = torch.tensor([[-0.492, 0.1333, 0.4879, pi, 2*pi, 2.3562]])
        # Increments in the original poses for sampling random values on each axis
        self.ee_pose_incs = torch.tensor([[-0.3,  0.3],
                                          [-0.3,  0.3],
                                          [-0.3,  0.3],
                                          [-0.5,  0.5],
                                          [-0.5,  0.5],
                                          [-0.5,  0.5]])
        
        # Which robot apply the sampling poses
        self.apply_range = False



        # ---- Target poses ----
        self.target_init = torch.tensor([[-0.4919, 0.1333, 0.4879, pi, 2*pi, 2.3562]])
        self.target_poses_incs = torch.tensor([[-0.075,  0.25],
                                               [-0.25,   0.25],
                                               [-0.3,   0.225],
                                               [-2*pi/5,  2*pi/5],
                                               [-2*pi/5,  2*pi/5],
                                               [-2*pi/5,  2*pi/5]])
        self.apply_range_tgt = True



        # --- Lie algebra ---
        # List of mappings
        map_list = [[[identity_map, identity_map], [exp_bruno, log_bruno], [exp_stereo, log_stereo]],
                [[identity_map, identity_map]],
                [[identity_map, identity_map]],
                [[identity_map, identity_map]],
                [[identity_map, identity_map]]]

        # List of conversions
        conversions = [[convert_dq_to_Lab, dq_from_tr], 
                       [None, None], 
                       [None, None], 
                       [None, None], 
                       [None, None]]
        
        diff_operators = [dq_diff, None, None, None, None]
        mul_operators = [dq_mul, None, None, None, None]

        # List of interpolators
        interpolators = [ScLERP, None, None, None, None]

        # Lis of distance functions
        distances = [[dqLOAM_distance, geodesic_dist, double_geodesic_dist],
                     [None],
                     [None],
                     [None],
                     [None]]
        
        identities = [[1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                      [None],
                      [None],
                      [None],
                      [None]]

        # Assign the functions according to configuration
        self.exp = map_list[representation][mapping][0]                 # Exponential mapping
        self.log = map_list[representation][mapping][1]                 # Logarithmic mapping
        self.convert_to_Lab = conversions[representation][0]                # Conversion Lie group to IsaacLab representation
        self.convert_to_group = conversions[representation][1]              # Conversion IsaacLab representation to Lie group     
        self.interpolator = interpolators[representation]                   # Interpolator function
        self.dist_function = distances[representation][distance]        # Distance function
        self.diff_operator = diff_operators[representation]
        self.mul_operator = mul_operators[representation]

        self.distance_thres = 0.02
        self.bonus_tgt_reached = 30



        

    # Computes the combined reward
    def compute_reward(self):
        dist = self.dist_function(self.ee_pose, self.target_pose)      
                                                                                

        # Obtains wether the agent is approaching or not
        # mod = 2*(dist < self.prev_dist).int() - 1

        # Target reached flag
        target_reached = dist < self.distance_thres


        # ---- Distance reward ----
        # Reward for the approaching
        reward = -10 * dist

        # ---- Reward composition ----
        # Phase reward plus bonuses
        reward = reward +  target_reached * self.bonus_tgt_reached

        # Update previous distances
        # self.prev_dist = dist
        

        return reward.item(), target_reached.item()


    # Getter for the observations
    def get_observation(self):
        a = log_parallel_transport(log_fc = self.log, 
                                   diff_operator = self.diff_operator,
                                   x = self.target_pose,
                                   q = self.ee_pose)[0]
        if torch.any(torch.isnan(a)):
            print(self.target_pose)
            print(self.ee_pose)
            print("DIFF: ", self.diff_operator(self.target_pose, self.ee_pose))
            print("LOG: ", self.log(self.diff_operator(self.target_pose, self.ee_pose), debug = True))
            print(a)

        return log_parallel_transport(log_fc = self.log, 
                                   diff_operator = self.diff_operator,
                                   x = self.target_pose,
                                   q = self.ee_pose)[0].numpy()



    # Step function
    def step(self, action):
        action = torch.tensor(action).float().unsqueeze(0)
        
        action[:, :3] = torch.clamp(action[:, :3], -self.action_scaling[0], self.action_scaling[0])
        action[:, 3:] = torch.clamp(action[:, 3:], -self.action_scaling[1], self.action_scaling[1])

        self.ee_pose_lie = self.ee_pose_lie + action
        self.ee_pose = exp_parallel_transport(exp_fc = self.exp,
                                        mul_operator = self.mul_operator,
                                        x = self.target_pose,
                                        s = self.ee_pose_lie)
        
        reward, done = self.compute_reward()

        return self.get_observation(), reward, done, False, {}
        



    # Reset function
    def reset(self, seed = None, options = None):
        # Generate uniform noise in the increments range
        rand_offsets = (self.ee_pose_incs[:, 1] - self.ee_pose_incs[:, 0]) * torch.rand(1, self.size_lie) + self.ee_pose_incs[:, 0]
        ee_pose = self.ee_init + rand_offsets
        ee_pose_quat = quat_from_euler_xyz(ee_pose[:, 3], ee_pose[:, 4], ee_pose[:, 5])
        self.ee_pose = self.convert_to_group(ee_pose[:, :3], ee_pose_quat)
        self.ee_pose_lie = self.log(self.ee_pose)


        rand_offsets = (self.target_poses_incs[:, 1] - self.target_poses_incs[:, 0]) * torch.rand(self.size_lie) + self.target_poses_incs[:, 0]
        target_pose = self.target_init + rand_offsets
        target_pose_quat = quat_from_euler_xyz(target_pose[:, 3], target_pose[:, 4], target_pose[:, 5])
        self.target_pose = self.convert_to_group(target_pose[:, :3], target_pose_quat)
        self.target_pose_lie = self.log(self.target_pose)

        # self.prev_dist = self.dist_function(self.ee_pose, self.target_pose)

        return self.get_observation(), {}


    # Render function
    def render(self):
        return 


    # Close function: shutdowns the simulation
    def close(self):
        pass


    # Set seed
    def seed(self, seed=None): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
