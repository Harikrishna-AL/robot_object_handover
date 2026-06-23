import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from builtin_interfaces.msg import Duration

import numpy as np
from stable_baselines3 import PPO
import torch
import random
import copy
import time
from math import pi

from scipy.spatial.transform import Rotation


class ActionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("action_node")


        # --- Parameters ---
        self.declare_parameter("model_path")
        self.declare_parameter("cmd")
        
        # model_8160000_steps
        self.model_path = "/home/docker_isaac_lab/IsaacLab/IsaacLab/source/logs/sb3/Isaac-UR5e-joint-reach-v0/2024-10-16_12-32-25/model_16680000_steps.zip"
        self.cmd = -2.3


        # --- Model ---
        self.model = PPO.load(self.model_path, weights_only=True)
        self.model.policy.eval()


        # --- Publishers ---
        # Publisher for the joint commands
        self.action_pub = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            10
        )


        # --- Subscribers ---
        # Subscriber for the observations
        __ = self.create_subscription(
            Float32MultiArray,
            "/aurova/observations",
            self.observation_cb,
            10
        )


        # --- Messages ---
        # Data observation from observation node
        self.obs_data = torch.tensor([0.0, 0.0], dtype=torch.float32)

        # Command / Objective for the agent
        self.euler_angles = [-1.57, -0.01, self.cmd]
        self.rot_quat = self.euler2quat(angles = self.euler_angles)
        self.command = torch.tensor([0.8172, 0.2329, 0.0628], dtype=torch.float32)
        self.command = torch.cat((self.command, self.rot_quat))

        # Previous action
        self.prev_action = torch.tensor([0.0], dtype=torch.float32)

        # Joint action message for the robot
        self.joint_traj_msg = Float64MultiArray()
        self.joint_traj_msg.data = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0]).tolist()

        # Reset flag
        self.reset = False
        

        # --- Timer ---
        self.timer = self.create_timer(2.0 / 50.0, self.timer_cb)


    # Convert Euler to Quaternion ('w' has to be at the end according to the IsaacLab agent)
    def euler2quat(self, angles):
        
        # Converts euler to quaternion
        rot = Rotation.from_euler(angles = angles, seq='xyz', degrees=False)
        quat = torch.tensor(rot.as_quat(), dtype=torch.float32)

        # Swap the first and last
        aux = copy.deepcopy(quat[0])
        quat[0] = quat[-1]
        quat[-1] = aux

        return quat


    # Converts Quaternion to Euler
    def quat2euler(self, quat):
        rot = Rotation.from_quat(quat)
        return torch.tensor(rot.as_euler('xyz', degrees=False), dtype=torch.float32)
    

    # Callback for the observation topic
    def observation_cb(self, msg):
        # Obtain the message as a tensor
        self.obs_data = torch.tensor(msg.data, dtype=torch.float32)

        # Convert to the range [0, 2*pi]
        cmd = copy.deepcopy(self.cmd)
        if cmd >= pi: cmd -= 2*pi
        cmd = cmd + 2*pi*int(cmd < 0.0)

        # Convert to the range [0, 2*pi] and to the IsaacLab frame (subvstract -2.3)
        pos = copy.deepcopy(self.obs_data[0])
        pos -= 2.3
        if pos <= -pi: pos += 2*pi
        pos = pos + 2*pi*int(pos < 0.0)

        # Set the reset flag
        self.reset = abs(pos - cmd)  < 0.03 or self.reset

        # Destroy node
        if self.reset:
            raise


    # Timer callback
    def timer_cb(self):
        
        # Do not use gradients in the predictions
        with torch.inference_mode():
            # Take the robot to home position
            if False: #self.steps >= 60:
                self.get_logger().info("RESETTING")

                # Reset joint actions
                self.joint_traj_msg.data = np.array([0.0, -0.0, 0.0, 0.0, 0.0, 0.0]).tolist()
                self.action_pub.publish(self.joint_traj_msg)

                # Re-load model
                self.model = PPO.load(self.model_path, weights_only=True)
                self.model.policy.eval()

                # Reset
                if self.steps >= 100:
                    
                    # Reset previous action
                    self.prev_action = torch.tensor([0.0], dtype=torch.float32)

                    # Generate new command
                    self.cmd = random.uniform(-1.0, 1.0)
                    self.euler_angles = [-1.57, -0.01, self.cmd]
                    self.rot_quat = self.euler2quat(angles = self.euler_angles)
                    self.command = torch.tensor([0.8172, 0.2329, 0.0628])
                    self.command = torch.cat((self.command, self.rot_quat))

                    
            # Apply action
            else:
                # Obtain obsevations
                full_obs = self.preprocess_obs()

                # Generate actions
                action, _states = self.model.predict(full_obs)

                # Apply scaled actions
                self.joint_traj_msg.data[-1] = action.item() * 0.5
                
                # If there is no reset, keep applying actions
                if not self.reset:
                    self.action_pub.publish(self.joint_traj_msg)
                    self.get_logger().info("--- " + str(self.joint_traj_msg.data[-1]))

                # Update previous actions
                self.prev_action = torch.tensor(action.squeeze(0), dtype=torch.float32)


    # Pre-process all the observation info
    def preprocess_obs(self):
        
        full_obs = torch.cat((self.obs_data, self.command, self.prev_action)).unsqueeze(dim=0)
        return full_obs
        


# --- Main / Entry Point ---
def main():
    rclpy.init()

    action_node = ActionNode()

    time.sleep(2)

    rclpy.spin(action_node)


if __name__ == "__main__":
    main()