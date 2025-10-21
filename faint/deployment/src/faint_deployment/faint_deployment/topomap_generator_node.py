'''
Listen to messages played back from a rosbag,
and sample images to generate a topological map.
'''

# ROS
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Message types
from sensor_msgs.msg import Image

# General
import cv2
import yaml
import time

from pathlib import Path
from threading import Lock

class TopomapGeneratorNode(Node):

    rgb_suffix = ".jpg"
    depth_suffix = "_depth.tiff"

    def __init__(self):
        super().__init__('create_topomap')

        # Ros parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('robot_config_path', rclpy.Parameter.Type.STRING),
                ('robot', rclpy.Parameter.Type.STRING),
                ('topomap_directory', rclpy.Parameter.Type.STRING),
                ('route_name', rclpy.Parameter.Type.STRING),
                ('dt', rclpy.Parameter.Type.DOUBLE),
                ('depth_rgb_sync_slop', rclpy.Parameter.Type.DOUBLE),
                ('depth_rgb_sync_queue_size', rclpy.Parameter.Type.INTEGER),
                ('namespace', rclpy.Parameter.Type.STRING),
                ('camera_type', rclpy.Parameter.Type.STRING),
                ('obs_mode', rclpy.Parameter.Type.STRING),
            ]
        )
        robot_config_path = self.get_parameter('robot_config_path').get_parameter_value().string_value
        robot = self.get_parameter('robot').get_parameter_value().string_value
        topomap_directory = self.get_parameter('topomap_directory').get_parameter_value().string_value
        route_name = self.get_parameter('route_name').get_parameter_value().string_value
        self.obs_mode = self.get_parameter('obs_mode').get_parameter_value().string_value
        assert self.obs_mode in ['rgb', 'depth', 'rgbd'], "obs_mode must be one of 'rgb', 'depth', 'rgbd'"

        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        assert self.dt > 0, "dt must be positive"

        robot_config_path = Path(robot_config_path)
        topomap_directory = Path(topomap_directory)

        with robot_config_path.open(mode="r", encoding="utf-8") as f:
            robot_configs = yaml.safe_load(f)
        robot_config = robot_configs[robot]

        # Remove the topomap directory if it already exists
        self.topomap_name_dir = topomap_directory / route_name
        if not self.topomap_name_dir.is_dir():
            self.topomap_name_dir.mkdir(parents=True)
        else:
            self.get_logger().info(f"{self.topomap_name_dir} already exists. Removing previous images...")
            for f in self.topomap_name_dir.iterdir():
                f.unlink()

        self._setup_ros(robot_config)
        camera = self.get_parameter('camera_type').get_parameter_value().string_value

        if self.obs_mode == 'rgb':
            cb_topics = [robot_config[camera]['camera_topic']]
        elif self.obs_mode == 'depth':
            cb_topics = [robot_config[camera]['depth_camera_topic']]
        elif self.obs_mode == 'rgbd':
            cb_topics = [robot_config[camera]['camera_topic'], robot_config[camera]['depth_camera_topic']]

        self.get_logger().info(f"Waiting for images on topics {cb_topics}")

    def _setup_ros(self, robot_config):

        self.bridge = CvBridge()

        # The messages are published from the bag under a namespace to avoid conflicts with existing topics
        dummy_namespace = self.get_parameter('namespace').get_parameter_value().string_value

        camera = self.get_parameter('camera_type').get_parameter_value().string_value

        ## Subscribers
        self.obs_lock = Lock()

        if self.obs_mode == 'rgb':
            self.rgb_img_msg = None
            self.rgb_sub = self.create_subscription(
                Image,
                dummy_namespace+robot_config[camera]["camera_topic"],
                self.rgb_callback,
                10,
            )
        elif self.obs_mode == 'depth':
            self.depth_img_msg = None
            self.depth_sub = self.create_subscription(
                Image,
                dummy_namespace+robot_config[camera]["depth_camera_topic"],
                self.depth_callback,
                10,
            )
        elif self.obs_mode == 'rgbd':
            self.rgb_img_msg = None
            self.depth_img_msg = None
            self.tss = ApproximateTimeSynchronizer(
                (
                    Subscriber(self, Image, dummy_namespace+robot_config[camera]["camera_topic"]),
                    Subscriber(self, Image, dummy_namespace+robot_config[camera]["depth_camera_topic"]),
                ),
                queue_size=self.get_parameter('depth_rgb_sync_queue_size').get_parameter_value().integer_value,
                slop=self.get_parameter('depth_rgb_sync_slop').get_parameter_value().double_value,
            )
            self.tss.registerCallback(self.rgbd_callback)

        ## Timer loop
        self.write_idx = 0
        self.latest_message_time = float("inf")
        self.timer = self.create_timer(
            self.dt,
            self.write_loop,
            callback_group = ReentrantCallbackGroup(),
        )

    def rgbd_callback(self, rgb_msg, depth_msg):
        self.get_logger().info("Received images")

        with self.obs_lock:
            self.rgb_img_msg = rgb_msg
            self.depth_img_msg = depth_msg

        self.latest_message_time = time.time()

    def rgb_callback(self, msg):
        self.get_logger().info("Received RGB image")
        with self.obs_lock:
            self.rgb_img_msg = msg
        self.latest_message_time = time.time()

    def depth_callback(self, msg):
        self.get_logger().info("Received depth image")
        with self.obs_lock:
            self.depth_img_msg = msg
        self.latest_message_time = time.time()


    def write_loop(self):

        self.get_logger().info("Checking for new images...")

        # If no new images received for 5 seconds, shut down
        if time.time() - self.latest_message_time > 5:
            self.get_logger().info(f"Subscribed topics not publishing anymore. Shutting down...")
            raise SystemExit

        if self.obs_mode == 'rgb':
            if self.rgb_img_msg is not None:
                with self.obs_lock:
                    rgb_img = self.bridge.imgmsg_to_cv2(self.rgb_img_msg, desired_encoding="bgr8")
                    self.rgb_img_msg = None
                cv2.imwrite(str(self.topomap_name_dir / f"{str(self.write_idx) + self.rgb_suffix}"), rgb_img)
                self.get_logger().info(f"Wrote topomap image {self.write_idx}")
                self.write_idx += 1

        elif self.obs_mode == 'depth':
            if self.depth_img_msg is not None:
                with self.obs_lock:
                    depth_img = self.bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="passthrough")
                    self.depth_img_msg = None
                cv2.imwrite(str(self.topomap_name_dir / f"{str(self.write_idx) + self.depth_suffix}"), depth_img)
                self.get_logger().info(f"Wrote topomap image {self.write_idx}")
                self.write_idx += 1

        elif self.obs_mode == 'rgbd':
            if self.rgb_img_msg is not None and self.depth_img_msg is not None:
                with self.rgb_img_msg_lock:
                    rgb_img = self.bridge.imgmsg_to_cv2(self.rgb_img_msg, desired_encoding="bgr8")
                    self.rgb_img_msg = None

                with self.depth_img_msg_lock:
                    depth_img = self.bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="passthrough")
                    self.depth_img_msg = None
                    
                cv2.imwrite(str(self.topomap_name_dir / f"{str(self.write_idx) + self.rgb_suffix}"), rgb_img)
                cv2.imwrite(str(self.topomap_name_dir / f"{str(self.write_idx) + self.depth_suffix}"), depth_img)

                self.get_logger().info(f"Wrote topomap image {self.write_idx}")
                self.write_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = TopomapGeneratorNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
