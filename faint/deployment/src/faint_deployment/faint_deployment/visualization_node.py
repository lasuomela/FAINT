'''
Publish data for visualization in Foxglove.
'''

import cv2
from pathlib import Path

# ROS
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# Message types
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from faint_interfaces.msg import PlaceRecognitionResult

# Package
from faint_deployment.policies.subgoal_selection.gallery_db import SubgoalDBHandler

class VisualizationNode(Node):

    '''
    Listen to the place recognition result and retrieve the
    corresponding topomap RGB and depth images from the database.
    Publish the images for visualization in Foxglove.
    '''

    def __init__(self):
        super().__init__('visualization_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('topomap_directory', rclpy.Parameter.Type.STRING),
                ('route_name', rclpy.Parameter.Type.STRING),
                ('subgoal_selection_model', rclpy.Parameter.Type.STRING),
                ('subgoal_lookahead', rclpy.Parameter.Type.INTEGER),
            ]
        )

        self.subgoal_lookahead = self.get_parameter('subgoal_lookahead').get_parameter_value().integer_value
        self.bridge = CvBridge()

        # Parse the topomap image directory
        self.topomap_img_dir = (
            Path(self.get_parameter('topomap_directory').get_parameter_value().string_value) /
                self.get_parameter('route_name').get_parameter_value().string_value
        )

        # Load the image gallery databases
        self.depth_img_db_handler = SubgoalDBHandler(self.topomap_img_dir, 'depth')
        self.rgb_img_db_handler = SubgoalDBHandler(self.topomap_img_dir, 'rgb')

        depth_images = self.depth_img_db_handler.get_images()
        rgb_images = self.rgb_img_db_handler.get_images()

        if depth_images and rgb_images:
            assert len(depth_images) == len(rgb_images), "RGB and depth image databases have different lengths"

        self.db_length = len(depth_images) if depth_images else len(rgb_images)

        # Serialize the images into ros messages
        self.rgb_img_msgs = []
        if rgb_images:
            for img in rgb_images:
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                rgb_img_msg = self.bridge.cv2_to_imgmsg(bgr_img, "bgr8")
                self.rgb_img_msgs.append(rgb_img_msg)

        self.depth_img_msgs = []
        if depth_images:
            for img in depth_images:
                depth_img_msg = self.bridge.cv2_to_imgmsg(img)
                self.depth_img_msgs.append(depth_img_msg)

        self._setup_ros()
        self.get_logger().info("Visualization node initialized")


    def _setup_ros(self):
        """
        Setup ROS publishers and subscribers
        """

        ## Subscribers

        # Place recognition subscriber
        self.place_recognition_sub = self.create_subscription(
            PlaceRecognitionResult,
            'closest_node_idx',
            self.place_recognition_callback,
            1,
        )

        # Goal reached subscriber
        self._goal_reached_sub = self.create_subscription(
            Bool,
            '/goal_reached',
            self._goal_reached_callback,
            10,
            callback_group = MutuallyExclusiveCallbackGroup(),
        )

        ## Publishers
        self.topomap_rgb_pub = self.create_publisher(Image, 'topomap_rgb', 10)
        self.topomap_depth_pub = self.create_publisher(Image, 'topomap_depth', 10)

        # Publish static transforms once at startup
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.make_transforms()

        self.get_logger().info("ROS setup complete")

    def make_transforms(self):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'

        self.tf_static_broadcaster.sendTransform(t)

    def _goal_reached_callback(self, msg):
        goal_reached = msg.data
        if goal_reached:
            self.get_logger().info("Goal reached, shutting down...")
            rclpy.shutdown()

    def place_recognition_callback(self, msg):
        """
        Publish the topomap images corresponding to the place recognition result
        """
        idx = msg.place_recognition_idx.data + self.subgoal_lookahead
        idx = min(idx, self.db_length - 1)

        if self.rgb_img_msgs:
            rgb_img_msg = self.rgb_img_msgs[idx]
            self.topomap_rgb_pub.publish(rgb_img_msg)

        if self.depth_img_msgs:
            depth_img_msg = self.depth_img_msgs[idx]
            self.topomap_depth_pub.publish(depth_img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == '__main__':
    main()
