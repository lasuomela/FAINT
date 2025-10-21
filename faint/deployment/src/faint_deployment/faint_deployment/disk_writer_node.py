"""
A ROS2 Node that subscribes to the observations and subgoal images,
and optionally overlays the predicted waypoints on the observations.
Both observation and subgoal image streams are saved to as video files.
"""
from typing import Tuple

import numpy as np
import yaml
from threading import Lock
from pathlib import Path
import cv2
from copy import deepcopy

# ROS
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Message types
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def project_waypoint(
        wp: Point,
        camera_height: float,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Tuple[int, int]:
    '''
    Project waypoint to image plane using camera intrinsics and distortion coefficients
    '''
    # Add the height of camera from ground and flip axes to opencv convention (x right, y down, z forward from image plane)
    wp_3d = np.expand_dims(np.array([-wp.y, camera_height, wp.x], dtype=np.float64), axis=0)
    t = np.array([0,0,0], dtype=np.float64)
    R = t
    wp_image, J = cv2.projectPoints(wp_3d, R, t, K, D)
    wp_image = wp_image.squeeze().astype(int)
    return wp_image[0], wp_image[1]

class DiskWriterNode(Node):

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
                ('robot_config_path', rclpy.Parameter.Type.STRING),
                ('robot', rclpy.Parameter.Type.STRING),
                ('camera_type', rclpy.Parameter.Type.STRING),
                ('topomap_directory', rclpy.Parameter.Type.STRING),
                ('route_name', rclpy.Parameter.Type.STRING),
            ]
        )

        self.bridge = CvBridge()
        self.obs_type = 'rgb'
        self.fps = 10

        # Load the robot config
        robot_config_path = Path(self.get_parameter('robot_config_path').get_parameter_value().string_value)
        with robot_config_path.open(mode="r", encoding="utf-8") as f:
            robot_configs = yaml.safe_load(f)
        self.robot_config = robot_configs[self.get_parameter('robot').get_parameter_value().string_value]

        self._setup_video_writers()
        self._setup_ros()
        self.get_logger().info("Disk writer node initialized")

    def goal_loop(self):
        """
        Loop for writing the goal images
        """
        # Save the subgoal image
        if self.waypoint_marker_msg is not None:
            with self._subgoal_img_msg_lock:
                if self.subgoal_img_msg is not None:
                    subgoal_img_msg = deepcopy(self.subgoal_img_msg)
            subgoal_img = self.bridge.imgmsg_to_cv2(subgoal_img_msg, self.image_encoding)
            self.subgoal_img_writer.write(subgoal_img)

    def obs_loop(self):
        """
        Loop for writing the observation images
        """
        # Save the observation image
        if self.waypoint_marker_msg is not None:

            with self._obs_img_msg_lock:
                obs_img_msg = deepcopy(self.obs_img_msg)

            if (obs_img_msg is not None) and (self.subgoal_img_msg is not None):
                obs_img = self.bridge.imgmsg_to_cv2(obs_img_msg, self.image_encoding)
                if self.waypoint_marker_msg is not None:
                    # Draw the waypoints on the observation image
                    obs_img = self._overlay_waypoints(obs_img, self.waypoint_marker_msg)

                self.obs_img_writer.write(obs_img)
        

    def _setup_video_writers(self):
        """
        Set up the video writers for the observation and subgoal images
        """
        # Set up the video writers
        obs_video_path = Path(self.get_parameter('topomap_directory').get_parameter_value().string_value) / f"{self.get_parameter('route_name').get_parameter_value().string_value}_obs.mp4"
        subgoal_video_path = Path(self.get_parameter('topomap_directory').get_parameter_value().string_value) / f"{self.get_parameter('route_name').get_parameter_value().string_value}_subgoal.mp4"

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.obs_img_writer = cv2.VideoWriter(str(obs_video_path), fourcc, self.fps, (1280, 720))
        self.subgoal_img_writer = cv2.VideoWriter(str(subgoal_video_path), fourcc, self.fps, (1280, 720))

    def _overlay_waypoints(self, img, waypoint_msg):
        """
        Overlay the waypoints on the observation image
        """
        waypoints_projected = []
        # Project the waypoints onto the image coords
        for waypoint in waypoint_msg.points:
            x, y = project_waypoint(waypoint, self.robot_config['camera_height'], self.cam_k, self.cam_d)
            waypoints_projected.append((x, y))
        
        # Add point close to the origo as the first waypoint
        # and project it onto the image plane
        pt = project_waypoint(Point(x=0.05, y=0.0, z=0.0), self.robot_config['camera_height'], self.cam_k, self.cam_d)
        waypoints_projected = [pt] + waypoints_projected

        # Plot the waypoints as lines
        x_prev, y_prev = None, None
        for x, y in waypoints_projected:
            try:
                if x_prev is not None and y_prev is not None:
                    cv2.line(img, (x_prev, y_prev), (x, y), (255, 0, 0), 6)
            except:
                raise Exception(f"type x: {type(x)}, type y: {type(y)}")
            x_prev, y_prev = x, y

        return img


    def __del__(self):
        self.get_logger().info("Disk writer node shutting down.")

        if hasattr(self, 'obs_timer'):
            self.obs_timer.cancel()
        if hasattr(self, 'goal_timer'):
            self.goal_timer.cancel()
        if hasattr(self, 'obs_img_writer'):
            with self._obs_img_msg_lock:
                self.obs_img_writer.release()
        if hasattr(self, 'subgoal_img_writer'):
            with self._subgoal_img_msg_lock:
                self.subgoal_img_writer.release()
        rclpy.shutdown()

    def _setup_ros(self):
        """
        Set up ROS publishers and subscribers
        """

        ## Subscribers

        # Observation image and place recognition result synchronizer
        self.obs_img_msg = None
        self.subgoal_img_msg = None
        self.goal_reached_msg = None
        self.camera_info_msg = None
        self.cam_k = None
        self.cam_d = None
        self.waypoint_marker_msg = None
        self._obs_img_msg_lock = Lock()
        self._subgoal_img_msg_lock = Lock()

        camera = self.get_parameter('camera_type').get_parameter_value().string_value
        if self.obs_type == "rgb":
            obs_topic = self.robot_config[camera]["camera_topic"]
            camera_info_topic = self.robot_config[camera]["camera_info_topic"]
            self.image_encoding = "bgr8"
        elif self.obs_type == "depth":
            raise NotImplementedError("Depth observation not supported yet")

        # Set QoS to Best Effort
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Depth of the queue
        )

        # Observation image subscriber
        self._obs_img_sub = self.create_subscription(
            Image,
            obs_topic,
            self._obs_img_callback,
            qos_profile,
        )

        # Subgoal image subscriber
        self.subgoal_img_sub = self.create_subscription(
            Image,
            '/topomap_rgb',
            self._subgoal_img_callback,
            qos_profile,
        )

        # Camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            qos_profile,
        )

        # Waypoint marker subscriber
        self.waypoint_marker_sub = self.create_subscription(
            Marker,
            '/waypoints',
            self._waypoint_marker_callback,
            qos_profile,
        )

        # Goal reached subscriber
        self._goal_reached_sub = self.create_subscription(
            Bool,
            '/goal_reached',
            self._goal_reached_callback,
            10,
            callback_group = MutuallyExclusiveCallbackGroup(),
        )

        # Set up a timer loop to process images
        self.obs_timer = self.create_timer(
            1/self.fps,
            self.obs_loop,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.goal_timer = self.create_timer(
            1/self.fps,
            self.goal_loop,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info("ROS setup complete")

    def _obs_img_callback(self, msg):
        with self._obs_img_msg_lock:
            self.obs_img_msg = msg

    def _subgoal_img_callback(self, msg):
        with self._subgoal_img_msg_lock:
            self.subgoal_img_msg = msg

    def _camera_info_callback(self, msg):
        if self.camera_info_msg is None:
            self.camera_info_msg = msg
            self.cam_k = np.array(msg.k).reshape((3,3))
            self.cam_d = np.array(msg.d)

    def _waypoint_marker_callback(self, msg):
        self.waypoint_marker_msg = msg

    def _goal_reached_callback(self, msg):
        self.goal_reached_msg = msg
        if self.goal_reached_msg.data:
            self.__del__()

def main(args=None):
    rclpy.init(args=args)
    node = DiskWriterNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == '__main__':
    main()
