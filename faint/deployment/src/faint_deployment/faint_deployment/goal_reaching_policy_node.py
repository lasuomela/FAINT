"""
Goal Reaching Policy Node
This module defines a ROS2 node that uses a goal-reaching policy to navigate a robot
towards a specified goal. The node subscribes to camera observations, and goals from
a subgoal selection policy, and publishes velocity commands to the robot.
"""

import numpy as np
import yaml
import pprint
import time
from threading import Lock
from pathlib import Path
from copy import deepcopy

# ROS
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Message types
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from faint_interfaces.msg import PlaceRecognitionResult

from faint_deployment.policies.base_model import dynamic_load
from faint_deployment.policies import goal_reaching
from faint_deployment.policies.subgoal_selection.gallery_db import SubgoalDBHandler
from faint_deployment.utils import create_waypoint_viz_marker

class GoalReachingPolicyNode(Node):
    def __init__(self):
        super().__init__('goal_reaching_policy_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_config_path', rclpy.Parameter.Type.STRING),
                ('robot', rclpy.Parameter.Type.STRING),
                ('topomap_directory', rclpy.Parameter.Type.STRING),
                ('route_name', rclpy.Parameter.Type.STRING),
                ('model_config_path', rclpy.Parameter.Type.STRING),
                ('model_weight_dir', rclpy.Parameter.Type.STRING),
                ('subgoal_selection_model', rclpy.Parameter.Type.STRING),
                ('goal_reaching_policy', rclpy.Parameter.Type.STRING),
                ('device', rclpy.Parameter.Type.STRING),
                ('main_loop_frequency', rclpy.Parameter.Type.DOUBLE),
                ('obs_img_place_recognition_sync_queue_size', rclpy.Parameter.Type.INTEGER),
                ('obs_img_place_recognition_sync_slop', rclpy.Parameter.Type.DOUBLE),
                ('dry_run', rclpy.Parameter.Type.BOOL),
                ('subgoal_lookahead', rclpy.Parameter.Type.INTEGER),
                ('pr_gr_tight_coupling', rclpy.Parameter.Type.BOOL),
                ('depth_cutoff_value', rclpy.Parameter.Type.DOUBLE),
                ('camera_type', rclpy.Parameter.Type.STRING),
            ],
        )

        # Load the robot config
        robot_config_path = Path(self.get_parameter('robot_config_path').get_parameter_value().string_value)
        with robot_config_path.open(mode="r", encoding="utf-8") as f:
            robot_configs = yaml.safe_load(f)
        self.robot_config = robot_configs[self.get_parameter('robot').get_parameter_value().string_value]

        # Parse the topomap image directory
        self.topomap_img_dir = (
            Path(self.get_parameter('topomap_directory').get_parameter_value().string_value) /
                self.get_parameter('route_name').get_parameter_value().string_value
        )

        self._setup_goal_reaching_policy()
        self._setup_ros()

        self.pub_count = 0
        self.pub_warmup_count = 5

        self.get_logger().info("Goal reaching policy node initialized")


    def _setup_goal_reaching_policy(self):
        """
        Load the goal reaching policy and the topological map.
        """

        self.goal_reached = False

        # Load the policy config
        model_config_path = Path(self.get_parameter('model_config_path').get_parameter_value().string_value)
        with model_config_path.open(mode="r", encoding="utf-8") as f:
            model_conf = yaml.safe_load(f)
        conf = model_conf[self.get_parameter('goal_reaching_policy').get_parameter_value().string_value]
        
        self.obs_type = conf['model']['obs_type']
        if self.obs_type == "depth":
            self.openni_depth = conf['model']['openni_depth']
            self.depth_cutoff_value = int( self.get_parameter('depth_cutoff_value').get_parameter_value().double_value * 1000)

        # Parse the model weight ckpt
        conf['model']['checkpoint_path'] = Path(self.get_parameter('model_weight_dir').get_parameter_value().string_value) / conf['model']["checkpoint_path"]

        self.get_logger().info(f"Loading goal reaching policy with config:")
        self.get_logger().info(f"{pprint.pformat(conf, indent=4)}")

        # Load the goal reaching policy
        Model = dynamic_load(goal_reaching, conf['model']['name'])
        self.goal_reaching_policy = Model(
            conf['model'],
            self.get_parameter('device').get_parameter_value().string_value,
        )

        # Load the gallery database
        self.img_db_handler = SubgoalDBHandler(
            img_dir_path=self.topomap_img_dir,
            img_type=self.obs_type,
        )
        self.subgoal_lookahead = self.get_parameter('subgoal_lookahead').get_parameter_value().integer_value


    def _setup_ros(self):
        """
        Set up ROS publishers, subscribers and timer loops.
        """

        self._cv_bridge = CvBridge()

        ### Subscribers
        self.obs_img_msg = None
        self.place_recognition_msg = None
        self._obs_img_msg_lock = Lock()
        self._place_recognition_msg_lock = Lock()

        camera = self.get_parameter('camera_type').get_parameter_value().string_value
        if self.obs_type == "rgb":
            obs_topic = self.robot_config[camera]["camera_topic"]
            self.image_encoding = 'rgb8'
        elif self.obs_type == "depth":
            obs_topic = self.robot_config[camera]["depth_camera_topic"]
            self.image_encoding = 'passthrough'

        if self.get_parameter('pr_gr_tight_coupling').get_parameter_value().bool_value:
            # Use obs images with approximately same time stamp as the image with
            # which the place recognition result was computed
            self.tss = ApproximateTimeSynchronizer(
                [
                    Subscriber(self, Image, obs_topic),
                    Subscriber(self, PlaceRecognitionResult, '/closest_node_idx')
                ],
                queue_size=self.get_parameter('obs_img_place_recognition_sync_queue_size').get_parameter_value().integer_value,
                slop=self.get_parameter('obs_img_place_recognition_sync_slop').get_parameter_value().double_value,
            )
            self.tss.registerCallback(self._obs_img_place_recognition_callback)
        else:

            # Set QoS to Best Effort
            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1  # Depth of the queue
            )

            # Run place recognition and goal reaching policy asynchronously
            self._obs_img_sub = self.create_subscription(
                Image,
                obs_topic,
                self._obs_img_callback,
                qos_profile,
            )

            self._place_recognition_sub = self.create_subscription(
                PlaceRecognitionResult,
                '/closest_node_idx',
                self._place_recognition_callback,
                10,
            )

        # Goal reached subscriber
        self._goal_reached_sub = self.create_subscription(
            Bool,
            '/goal_reached',
            self._goal_reached_callback,
            10
        )

        ### Publishers
        self.cmd_vel_msg = None
        self.cmd_vel_msg_lock = Lock()
        cmd_vel_topic = self.robot_config["vel_navi_topic"]
        if self.get_parameter('dry_run').get_parameter_value().bool_value:
            cmd_vel_topic += "_dry_run"
            
        self._cmd_vel_pub = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )

        # Publisher for the waypoint visualization
        self._waypoint_pub = self.create_publisher(
            Marker,
            '/waypoints',
            10,
        )

        # Publisher for the runtime duration statistic
        self._runtime_pub = self.create_publisher(
            Float32,
            '/goal_reaching_policy_runtime',
            10,
        )

        # Set up a timer loop to process images
        self.main_loop_frequency = self.get_parameter('main_loop_frequency').get_parameter_value().double_value
        self.main_timer = self.create_timer(
            1/self.main_loop_frequency,
            self.main_loop,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Set up a timer loop to publish velocity commands
        self.cmd_timer = self.create_timer(
            1/self.robot_config['cmd_loop_frequency'],
            self._cmd_vel_loop,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def _obs_img_callback(self, msg):
        with self._obs_img_msg_lock:
            self.obs_img_msg = msg
            
    def _place_recognition_callback(self, msg):
        with self._place_recognition_msg_lock:
            self.place_recognition_msg = msg

    def _obs_img_place_recognition_callback(self, obs_img_msg, place_recognition_msg):
        # Callback for synchronized obs image and place recognition result
        with self._obs_img_msg_lock and self._place_recognition_msg_lock:
            self.obs_img_msg = obs_img_msg
            self.place_recognition_msg = place_recognition_msg

    def _goal_reached_callback(self, msg):
        self.goal_reached = msg.data

        if self.goal_reached:
            self.get_logger().info("Goal reached, shutting down...")
            self.main_timer.cancel()
            self.cmd_timer.cancel()
            rclpy.shutdown()

    def main_loop(self):
        """
        Feed latest observation and subgoal image to the goal reaching policy
        to produce robot control commands.
        """

        start_time = time.time()

        # Get the place recognition result
        with self._place_recognition_msg_lock and self._obs_img_msg_lock:
            place_recognition_msg = self.place_recognition_msg
            if place_recognition_msg is None:
                return
            
            obs_img_msg = deepcopy(self.obs_img_msg)
            if obs_img_msg is None:
                return
            self.obs_img_msg = None

        # Get the goal image from the gallery database
        goal_idx = min(
            place_recognition_msg.place_recognition_idx.data + self.subgoal_lookahead,
            self.img_db_handler.gallery_len - 1
        )
        closest_db_img = self.img_db_handler.get_image_by_idx(
            goal_idx,
        )

        # Get the obs image from the message
        obs_img = self._cv_bridge.imgmsg_to_cv2(
            obs_img_msg,
            desired_encoding=self.image_encoding,
        )

        # Preprocess depth images
        if self.obs_type == "depth":
            obs_img = self.prepare_depth_image(obs_img)
            closest_db_img = self.prepare_depth_image(closest_db_img)

        # Prepare the data dictionary for the policy
        data_dict = {
            'obs': obs_img,
            'goal': closest_db_img,
            'max_lin_vel': self.robot_config['max_v'],
            'max_ang_vel': self.robot_config['max_w'],
            'main_loop_frequency': self.main_loop_frequency,
        }

        # Get the action from the policy
        action = self.goal_reaching_policy(data_dict)

        # Do not publish the first self.pub_warmup_count
        # messages because there is some overhead in the first few inferences
        if self.pub_count < self.pub_warmup_count:
            self.pub_count += 1
        else:
            # Build the action message for publishing
            twist_msg = Twist()
            twist_msg.linear.x = action['v']
            twist_msg.angular.z = action['w']
            with self.cmd_vel_msg_lock:
                self.cmd_vel_msg = twist_msg

            # Publish waypoints for visualization
            if 'waypoints' in action:
                waypoints = action['waypoints']
                # Create a line strip marker
                marker = create_waypoint_viz_marker(waypoints, self.get_clock().now().to_msg())
                self._waypoint_pub.publish(marker)

        # Publish the runtime duration
        runtime_msg = Float32()
        runtime_msg.data = time.time() - start_time
        self._runtime_pub.publish(runtime_msg)
    
    def _cmd_vel_loop(self):
        """
        Publish the most recent velocity command.
        """

        with self.cmd_vel_msg_lock:
            cmd_vel_msg = self.cmd_vel_msg

        if cmd_vel_msg is not None:
            assert cmd_vel_msg.linear.x <= self.robot_config['max_v'] +0.1, f"{cmd_vel_msg.linear.x} > {self.robot_config['max_v']}"
            assert cmd_vel_msg.angular.z <= self.robot_config['max_w'] +0.1, f"{cmd_vel_msg.angular.z} > {self.robot_config['max_w']}"
            self._cmd_vel_pub.publish(cmd_vel_msg)

    def prepare_depth_image(self, depth_img):
        """
        Prepare the depth images for the policy.
        """
        depth_img[depth_img > self.depth_cutoff_value] = self.depth_cutoff_value

        if not self.openni_depth:
            multiplier = (2**16 -1) / self.depth_cutoff_value
            depth_img = (depth_img * multiplier).round().astype(np.uint16)
        return depth_img


def main(args=None):
    rclpy.init(args=args)

    goal_reaching_policy_node = GoalReachingPolicyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(goal_reaching_policy_node)
    executor.spin()

    goal_reaching_policy_node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()


