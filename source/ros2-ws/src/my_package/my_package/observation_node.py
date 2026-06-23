import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


class ObservationNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("observation_node")

        # --- Publishers ---
        # Publisher for broadcasting the observations from the environment
        self.observation_pub = self.create_publisher(
            Float32MultiArray,
            "/aurova/observations",
            10
        )

        # --- Subscribers ---
        # Subscriber for the joint_states
        __ = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_cb,
            10
        )

        # --- Timer ---

        # --- Messages ---
        self.msg = Float32MultiArray()
        self.msg.data = [0.0, 0.0]


    # Joint state callback
    # Gathers information about the joints
    def joint_states_cb(self, msg):
        idx = msg.name.index("wrist_3_joint")
        self.msg.data[0] = msg.position[idx]
        self.msg.data[1] = msg.velocity[idx]
        self.observation_pub.publish(self.msg)




# --- Main / EntryPoint ---
def main():
    rclpy.init()
    obs_node = ObservationNode()

    rclpy.spin(obs_node)


if __name__ == "__main__":
    main()
