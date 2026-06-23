#!/usr/bin/env python3
import sys
import os
# from stable_baselines import PPO
import rclpy
import rclpy.node


class My_node(rclpy.node.Node):
    def __init__(self):
        super().__init__('node')

        self.declare_parameter("model_path")
        self.timer = self.create_timer(2.0 / 50.0, self.timer_callback)

    def timer_callback(self):
        param = self.get_parameter("model_path").get_parameter_value().string_value
        self.get_logger().info(param)


def main():
    rclpy.init()
    node = My_node()
    rclpy.spin(node)

if __name__ == "__main__":
    main() 
