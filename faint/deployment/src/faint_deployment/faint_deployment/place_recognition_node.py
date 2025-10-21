"""
This module defines the PlaceRecognitionNode class, which is a ROS2
node for performing place recognition using various models and filters.
The node subscribes to image topics, processes the images to recognize places,
and publishes the results.
"""
from typing import Dict

import yaml
import time
import torch
from pathlib import Path
from threading import Lock

# ROS
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

# Message types
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32, Float32
from faint_interfaces.msg import PlaceRecognitionResult

# Place recognition
from faint_deployment.policies.subgoal_selection.bayesian_querier import PlaceRecognitionTopologicalFilter
from faint_deployment.policies.subgoal_selection.sliding_window_querier import PlaceRecognitionSlidingWindowFilter
from faint_deployment.policies.subgoal_selection.tempdist_sliding_window_querier import TemporalDistanceSlidingWindowFilter
from faint_deployment.policies.subgoal_selection.feature_extractor import FeatureExtractor
from faint_deployment.policies.subgoal_selection.gallery_db import SubgoalDBHandler
from faint_deployment.policies.subgoal_selection import extract_database
from faint_deployment.utils import get_image_transform

# Temporal distance
from faint_deployment.policies.goal_reaching.vint.utils import load_model

class PlaceRecognitionNode(Node):

    def __init__(self):
        super().__init__('place_recognition_node')
        self.logger = self.get_logger()

        self.declare_parameters(
            namespace='',
            parameters=[
                ('topomap_directory', rclpy.Parameter.Type.STRING),
                ('route_name', rclpy.Parameter.Type.STRING),
                ('model_config_path', rclpy.Parameter.Type.STRING),
                ('model_weight_dir', rclpy.Parameter.Type.STRING),
                ('subgoal_selection_model', rclpy.Parameter.Type.STRING),
                ('filter_mode', rclpy.Parameter.Type.STRING),
                ('sliding_window_radius', rclpy.Parameter.Type.INTEGER),
                ('transition_model_window_lower', rclpy.Parameter.Type.INTEGER),
                ('transition_model_window_upper', rclpy.Parameter.Type.INTEGER),
                ('bayesian_filter_delta', rclpy.Parameter.Type.INTEGER),
                ('recompute_db', rclpy.Parameter.Type.BOOL),
                ('device', rclpy.Parameter.Type.STRING),
                ('main_loop_frequency', rclpy.Parameter.Type.DOUBLE),
                ('start_node_idx', rclpy.Parameter.Type.INTEGER),
                ('goal_node_idx', rclpy.Parameter.Type.INTEGER),
                ('robot_config_path', rclpy.Parameter.Type.STRING),
                ('robot', rclpy.Parameter.Type.STRING),
                ('camera_type', rclpy.Parameter.Type.STRING),
            ]
        )

        self._img_lock = Lock()
        self._latest_img_msg = None
        self._first_pass = True

        # Set frequently accessed parameters as class attributes
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.filter_mode = self.get_parameter('filter_mode').get_parameter_value().string_value
        self.window_radius = self.get_parameter('sliding_window_radius').get_parameter_value().integer_value
        self.closest_node_idx = self.get_parameter('start_node_idx').get_parameter_value().integer_value
        self.goal_node_idx = self.get_parameter('goal_node_idx').get_parameter_value().integer_value

        # Parse the topomap image directory
        self.topomap_img_dir = (
            Path(self.get_parameter('topomap_directory').get_parameter_value().string_value) /
                self.get_parameter('route_name').get_parameter_value().string_value
        )

        # Load the robot config
        robot_config_path = Path(self.get_parameter('robot_config_path').get_parameter_value().string_value)
        with robot_config_path.open(mode="r", encoding="utf-8") as f:
            robot_configs = yaml.safe_load(f)
        self.robot_config = robot_configs[self.get_parameter('robot').get_parameter_value().string_value]


        # Load the subgoal selection model config
        model_config_path = Path(self.get_parameter('model_config_path').get_parameter_value().string_value)
        subgoal_selection_model = self.get_parameter('subgoal_selection_model').get_parameter_value().string_value
        with model_config_path.open(mode="r", encoding="utf-8") as f:
            confs = yaml.safe_load(f)
            conf = confs[subgoal_selection_model]

        # Setup the subgoal selection model
        if conf['model']['type'] == 'temporal_distance':
            # GNM / ViNT / NoMAD style subgoal selection
            self._setup_temporal_distance(
                conf,
                Path(self.get_parameter('model_weight_dir').get_parameter_value().string_value),
                self.get_parameter('filter_mode').get_parameter_value().string_value,
            )
        elif conf['model']['type'] == 'place_recognition':  
            self._setup_place_recognition(
                conf,
                Path(self.get_parameter('model_weight_dir').get_parameter_value().string_value),
                subgoal_selection_model,
                self.get_parameter('filter_mode').get_parameter_value().string_value,
                self.get_parameter('transition_model_window_lower').get_parameter_value().integer_value,
                self.get_parameter('transition_model_window_upper').get_parameter_value().integer_value,
                self.get_parameter('bayesian_filter_delta').get_parameter_value().integer_value,
                self.get_parameter('recompute_db').get_parameter_value().bool_value,
            )
        else:
            raise ValueError(f"Model type {conf['model']['type']} not recognized")

        self._setup_ros()
        self.logger.info(
            f"""
            Place recognition node ready, waiting for images from topic 
            {self.robot_config[self.get_parameter('camera_type').get_parameter_value().string_value]['camera_topic']}
            """)

    
    def img_callback(self, msg):
        with self._img_lock:
            self._latest_img_msg = msg

    def main_loop(self):
        '''
        Choose the next subgoal based on the current observation
        using place recognition or temporal distance prediction,
        and optionally apply a filter to smooth the subgoal selection.
        '''
        start_time = time.time()
        with self._img_lock:
            img_msg = self._latest_img_msg
            self._latest_img_msg = None

        if img_msg is None:
            return
        
        current_obs = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        current_obs = self._image_transform(current_obs).unsqueeze(0)

        # Initialize the belief distribution of the
        # Bayesian filter prior to first query
        if (self._first_pass and
            self.filter_mode == 'bayesian'
            ):
            self.subgoal_querier.initialize_model(current_obs)
            self._first_pass = False
        
        # Place recognition with Bayesian filter
        if self.filter_mode == 'bayesian':
            self.closest_node_idx, score = self.subgoal_querier.match(current_obs)

        # Place recognition with sliding window filter
        elif self.filter_mode == 'sliding_window':
            start = max(self.closest_node_idx - self.window_radius, 0)
            end = min(self.closest_node_idx + self.window_radius + 1, self.goal_node_idx + 1)
            self.closest_node_idx = self.subgoal_querier.match(current_obs, start, end)

        else:
            raise ValueError(f"Filter mode {self.filter_mode} not recognized!")
        
        # Check if the goal has been reached
        reached_goal = self.closest_node_idx == self.goal_node_idx
        self.goal_pub.publish(Bool(data=reached_goal.item()))
        if reached_goal:
            self.logger.info("Reached goal. Stopping...")
            self.timer.cancel()
            rclpy.shutdown()
        else:
        
            # Publish the idx of the closest node
            pr_result = PlaceRecognitionResult(
                header=img_msg.header,
                place_recognition_idx=Int32(
                    data=self.closest_node_idx.item()
                ),
            )
            self.place_recognition_pub.publish(pr_result)

            # Publish the runtime duration
            runtime_msg = Float32()
            runtime_msg.data = time.time() - start_time
            self._runtime_pub.publish(runtime_msg)
        
    def _setup_ros(self):
        """
        Set up ROS2 publishers and subscribers
        """

        self.bridge = CvBridge()

        camera = self.get_parameter('camera_type').get_parameter_value().string_value

        # Subscribers
        # Set QoS to Best Effort
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Depth of the queue
        )
        self.image_sub = self.create_subscription(
            Image,
            self.robot_config[camera]["camera_topic"],
            self.img_callback,
            qos_profile,
        )

        # Publishers
        self.goal_pub = self.create_publisher(
            Bool,
            'goal_reached',
            1,
        )
        self.place_recognition_pub = self.create_publisher(
            PlaceRecognitionResult,
            'closest_node_idx',
            1,
        )

        self._runtime_pub = self.create_publisher(
            Float32,
            'place_recognition_runtime',
            1,
        )

        # Set up a timer loop to process images
        self.timer = self.create_timer(
            1 / self.get_parameter('main_loop_frequency').get_parameter_value().double_value,
            self.main_loop, 
            callback_group = MutuallyExclusiveCallbackGroup(),
        )

    def _setup_temporal_distance(
        self,
        model_config: Dict,
        model_weight_dir: Path,
        filter_mode: str,
        ):
        """
        Load a subgoal selection policy based on temporal distance prediction.
        """

        if 'checkpoint_path' in model_config['model']:
            model_config['model']["checkpoint_path"] = model_weight_dir / model_config['model']["checkpoint_path"]

        # Get the image preprocessing transform for the model
        self._image_transform = get_image_transform(model_config['model']['image_size'])

        # Set up the temporal distance prediction model
        if Path(model_config['model']['checkpoint_path']).suffix == '.pt':
            # TorchScript model trained by us
            extractor = torch.jit.optimize_for_inference(
                torch.jit.load(
                    model_config['model']['checkpoint_path'],
                    map_location=self.device,
                ),
            )
            extractor.eval()
        else:
            # Load the original ViNT model
            extractor = load_model(
                model_config['model']['checkpoint_path'],
                model_config['model'],
                self.device,
            )
            extractor.eval()

        # Set up the subgoal database handler
        db_handler = SubgoalDBHandler(
            img_dir_path=self.topomap_img_dir,
            img_type='rgb',
            image_transform=self._image_transform, # Precompute the image transform for the db images
        )

        # Init the place recognition querier
        if filter_mode == 'bayesian':
            raise ValueError("""Bayesian filter not supported for temporal distance prediction!
                             (Hint: launch with filter_mode:=sliding_window)""")
        elif filter_mode == 'sliding_window':
            self.subgoal_querier = TemporalDistanceSlidingWindowFilter(
                extractor=extractor,
                db_handler=db_handler,
                device=self.device,
                sequence_length=model_config['model']['context_size']+1,
                model_type=model_config['model']['model_type'],
            )
        else:
            raise ValueError(f"Filter mode {filter_mode} not recognized")
        
        # Check if the goal node index is valid
        map_size = self.subgoal_querier.get_map_size()
        assert -1 <= self.goal_node_idx < map_size, "Invalid goal index"
        if self.goal_node_idx == -1:
            self.goal_node_idx = map_size - 1

    def _setup_place_recognition(
            self,
            model_config: Dict,
            model_weight_dir: Path,
            subgoal_selection_model: str,
            filter_mode: str,
            transition_model_window_lower: int,
            transition_model_window_upper: int,
            bayesian_filter_delta: int,
            recompute_db: bool = False,
        ):
        """
        Load a subgoal selection policy based on place recognition.
        """

        if 'checkpoint_path' in model_config['model']:
            model_config['model']["checkpoint_path"] = model_weight_dir / model_config['model']["checkpoint_path"]

        # Get the image preprocessing transform for the model
        self._image_transform = get_image_transform(model_config['model']['image_size'])

        # Extract the global descriptors from the topomap images
        place_recognition_db_path = self.topomap_img_dir / f"global-feats-{subgoal_selection_model}.h5"
        if place_recognition_db_path.exists() and recompute_db:
            self.logger.info(f"Recomputing features from topomaps in {self.topomap_img_dir}")
            place_recognition_db_path.unlink()

        if not place_recognition_db_path.exists():
            self.logger.info(f"Extracting features from topomaps in {self.topomap_img_dir}")
            extract_database.main(
                model_config,
                self.topomap_img_dir,
                self._image_transform,
                self.topomap_img_dir,
                as_half=False,
                )
        else:
            self.logger.info(f"Found existing database at {place_recognition_db_path}")

        # Init the place recognition model
        extractor = FeatureExtractor(model_config, self.device)

        # Init the subgoal database handler
        db_handler = SubgoalDBHandler(
            img_dir_path=self.topomap_img_dir,
            img_type='rgb',
            db_path=place_recognition_db_path,
        )

        # Init the place recognition querier
        if filter_mode == 'bayesian':
            self.subgoal_querier = PlaceRecognitionTopologicalFilter(
                extractor=extractor,
                db_handler=db_handler,
                delta=bayesian_filter_delta,
                window_lower=transition_model_window_lower,
                window_upper=transition_model_window_upper,
                )   
        elif filter_mode == 'sliding_window':
            self.subgoal_querier = PlaceRecognitionSlidingWindowFilter(
                extractor=extractor,
                db_handler=db_handler,
            )
        else:
            raise ValueError(f"Filter mode {filter_mode} not recognized")
        
        # Check if the goal node index is valid
        map_size = self.subgoal_querier.get_map_size()
        assert -1 <= self.goal_node_idx < map_size, "Invalid goal index"
        if self.goal_node_idx == -1:
            self.goal_node_idx = map_size - 1
        
def main(args=None):
    rclpy.init(args=args)
    node = PlaceRecognitionNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == '__main__':
    main()
