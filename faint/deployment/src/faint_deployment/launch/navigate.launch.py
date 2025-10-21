from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
import launch_ros
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

# Get the python package top directory and make paths relative to it
# Note: This will only work properly if the package was installed as editable (pip install -e .)
# Other option would be to make paths relative to ros pkg share directory but would mean that the package
# have to be rebuilt after every change in the config files
from pathlib import Path
import faint
pkg_top_dir = Path(faint.__file__).parent

def generate_launch_description():
    # Declare the launch arguments
    declare_topomap_directory = DeclareLaunchArgument(
        'topomap_directory',
        default_value=str(pkg_top_dir / 'deployment/topomaps/images'),
        description='Directory where the topomap will be stored'
    )
    declare_model_config_path = DeclareLaunchArgument(
        'model_config_path',
        default_value=str(pkg_top_dir / 'deployment/src/faint_deployment/config/models.yaml'),
        description='Path to the model configuration file'
    )
    declare_model_weight_dir = DeclareLaunchArgument(
        'model_weight_dir',
        default_value=str(pkg_top_dir / 'deployment/src/faint_deployment/model_weights'),
        description='Directory where the model weights are stored'
    )
    declare_robot_config_path = DeclareLaunchArgument(
        'robot_config_path',
        default_value=str(pkg_top_dir / 'deployment/src/faint_deployment/config/robots.yaml'),
        description='Path to the robot configuration YAML file'
    )
    declare_route_name = DeclareLaunchArgument(
        'route_name',
        description='Name of the route for the topomap'
    )
    declare_subgoal_selection_model = DeclareLaunchArgument(
        'subgoal_selection_model',
        default_value='eigenplaces',
        description='Place recognition model name (hint: check config/models.yaml) (default: eigenplaces)',
    )
    declare_filter_mode = DeclareLaunchArgument(
        'filter_mode',
        default_value='bayesian',
        description='Filter mode ( bayesian | sliding_window )'
    )
    declare_sliding_window_radius = DeclareLaunchArgument(
        'sliding_window_radius',
        default_value='2',
        description="""Number of nodes to look at in the topopmap for
            localization when using sliding window filter (default: 2)"""
    )
    declare_transition_model_window_lower = DeclareLaunchArgument(
        'transition_model_window_lower',
        default_value='-1',
        description='Bayesian filter transition model window lower bound (default: -1)'
    )
    declare_transition_model_window_upper = DeclareLaunchArgument(
        'transition_model_window_upper',
        default_value='2',
        description='Bayesian filter transition model window upper bound (default: 2)'
    )
    declare_bayesian_filter_delta = DeclareLaunchArgument(
        'bayesian_filter_delta',
        default_value='10',
        description='Bayesian filter delta (default: 10)'
    )
    declare_recompute_db = DeclareLaunchArgument(
        'recompute_db',
        default_value='False',
        description='If the place recognition database should be recomputed (default: False)'
    )
    declare_device = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Compute device (default: cuda)'
    )
    declare_start_node_idx = DeclareLaunchArgument(
        'start_node_idx',
        default_value='0',
        description="""Start node index in the topomap (if 0, then the start node is 
        the first node in the topomap) (default: 0)"""
    )
    declare_goal_node_idx = DeclareLaunchArgument(
        'goal_node_idx',
        default_value='-1',
        description="""Goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)"""
    )
    declare_robot = DeclareLaunchArgument(
        'robot',
        default_value='turtlebot4',
        description='Name of the robot to use from the configuration file'
    )
    declare_camera_type = DeclareLaunchArgument(
        'camera_type',
        default_value='zed2',
        description='Camera type to use for the rgb image'
    )
    declare_img_pr_sync_queue_size = DeclareLaunchArgument(
        'img_pr_sync_queue_size',
        default_value='100',
        description='Queue size for synchronizing rgb observations and place recognition result',
    )
    declare_img_pr_sync_slop = DeclareLaunchArgument(
        'img_pr_sync_slop',
        default_value='0.03',
        description='Slop time in seconds for synchronizing rgb observations and place recognition result',
    )
    declare_place_recognition_frequency = DeclareLaunchArgument(
        'place_recognition_frequency',
        default_value='4.0',
        description='Frequency of the place recognition node in Hz'
    )
    declare_goal_reaching_policy_frequency = DeclareLaunchArgument(
        'goal_reaching_policy_frequency',
        default_value='4.0',
        description='Frequency of the goal reaching policy node in Hz'
    )
    declare_goal_reaching_policy = DeclareLaunchArgument(
        'goal_reaching_policy',
        default_value='FAINT-Sim',
        description='Goal reaching policy name (hint: check config/models.yaml) (default: FAINT-Sim)',
    )
    declare_subgoal_lookahead = DeclareLaunchArgument(
        'subgoal_lookahead',
        default_value='1',
        description='How many nodes ahead from the place recognition result to pick the subgoal (default: 1)'
    )
    declare_pr_gr_tight_coupling = DeclareLaunchArgument(
        'pr_gr_tight_coupling',
        default_value='False',
        description='If the place recognition and goal reaching policy should be run with observations from same timestamp (default: False)'
    )
    declare_depth_cutoff_value = DeclareLaunchArgument(
        'depth_cutoff_value',
        default_value='10.0',
        description='Maximum depth threshold (meters) for depth images passed to the goal reaching policy (default: 10.0)'
    )
    declare_dry_run = DeclareLaunchArgument(
        'dry_run',
        default_value='False',
        description='If the robot should move or not (default: False)'
    )
    declare_viz_condition = DeclareLaunchArgument(
        'visualize',
        default_value='True',
        description='If the visualization tools should be launched (default: False)'
    )
    declare_write_to_disk = DeclareLaunchArgument(
        'write_to_disk',
        default_value='False',
        description='If the observation and subgoal image streams should be written to disk (default: False)',
    )

    # Define the place recognition node with parameters
    place_recognition_node = Node(
        package='faint_deployment',
        executable='place_recognition_node',
        name='place_recognition',
        output='screen',
        parameters=[{
            'topomap_directory': LaunchConfiguration('topomap_directory'),
            'route_name': LaunchConfiguration('route_name'),
            'model_config_path': LaunchConfiguration('model_config_path'),
            'model_weight_dir': LaunchConfiguration('model_weight_dir'),
            'subgoal_selection_model': LaunchConfiguration('subgoal_selection_model'),
            'filter_mode': LaunchConfiguration('filter_mode'),
            'sliding_window_radius': LaunchConfiguration('sliding_window_radius'),
            'transition_model_window_lower': LaunchConfiguration('transition_model_window_lower'),
            'transition_model_window_upper': LaunchConfiguration('transition_model_window_upper'),
            'bayesian_filter_delta': LaunchConfiguration('bayesian_filter_delta'),
            'recompute_db': LaunchConfiguration('recompute_db'),
            'device': LaunchConfiguration('device'),
            'main_loop_frequency': LaunchConfiguration('place_recognition_frequency'),
            'start_node_idx': LaunchConfiguration('start_node_idx'),
            'goal_node_idx': LaunchConfiguration('goal_node_idx'),
            'robot_config_path': LaunchConfiguration('robot_config_path'),
            'robot': LaunchConfiguration('robot'),
            'camera_type': LaunchConfiguration('camera_type'),
        }]
    )

    goal_reaching_policy_node = Node(
        package='faint_deployment',
        executable='goal_reaching_policy_node',
        name='goal_reaching_policy',
        output='screen',
        parameters=[{
            'model_config_path': LaunchConfiguration('model_config_path'),
            'model_weight_dir': LaunchConfiguration('model_weight_dir'),
            'robot_config_path': LaunchConfiguration('robot_config_path'),
            'robot': LaunchConfiguration('robot'),
            'topomap_directory': LaunchConfiguration('topomap_directory'),
            'route_name': LaunchConfiguration('route_name'),
            'subgoal_selection_model': LaunchConfiguration('subgoal_selection_model'),
            'goal_reaching_policy': LaunchConfiguration('goal_reaching_policy'),
            'subgoal_lookahead': LaunchConfiguration('subgoal_lookahead'),
            'main_loop_frequency': LaunchConfiguration('goal_reaching_policy_frequency'),
            'obs_img_place_recognition_sync_queue_size': LaunchConfiguration('img_pr_sync_queue_size'),
            'obs_img_place_recognition_sync_slop': LaunchConfiguration('img_pr_sync_slop'),
            'device': LaunchConfiguration('device'),
            'dry_run': LaunchConfiguration('dry_run'),
            'pr_gr_tight_coupling': LaunchConfiguration('pr_gr_tight_coupling'),
            'depth_cutoff_value': LaunchConfiguration('depth_cutoff_value'),
            'camera_type': LaunchConfiguration('camera_type'),
        }]
    )

    # Include the visualization node if the visualize argument is set to True
    visualization_node = Node(
        package='faint_deployment',
        executable='visualization_node',
        name='visualization',
        output='screen',
        parameters=[{
            'topomap_directory': LaunchConfiguration('topomap_directory'),
            'route_name': LaunchConfiguration('route_name'),
            'subgoal_selection_model': LaunchConfiguration('subgoal_selection_model'),
            'subgoal_lookahead': LaunchConfiguration('subgoal_lookahead'),
        }],
        condition=IfCondition(LaunchConfiguration('visualize')),
    )

    # Include disk writer node if the write_to_disk argument is set to True
    disk_writer_node = Node(
        package='faint_deployment',
        executable='disk_writer_node',
        name='disk_writer',
        output='screen',
        parameters=[{
            'topomap_directory': LaunchConfiguration('topomap_directory'),
            'route_name': LaunchConfiguration('route_name'),
            'robot_config_path': LaunchConfiguration('robot_config_path'),
            'robot': LaunchConfiguration('robot'),
            'camera_type': LaunchConfiguration('camera_type'),
        }],
        condition=IfCondition(LaunchConfiguration('write_to_disk')),
    )

    return LaunchDescription([
        declare_topomap_directory,
        declare_route_name,
        declare_model_config_path,
        declare_model_weight_dir,
        declare_subgoal_selection_model,
        declare_filter_mode,
        declare_sliding_window_radius,
        declare_transition_model_window_lower,
        declare_transition_model_window_upper,
        declare_bayesian_filter_delta,
        declare_recompute_db,
        declare_device,
        declare_start_node_idx,
        declare_goal_node_idx,
        declare_robot_config_path,
        declare_robot,
        declare_camera_type,
        declare_img_pr_sync_queue_size,
        declare_img_pr_sync_slop,
        declare_place_recognition_frequency,
        declare_goal_reaching_policy_frequency,
        declare_goal_reaching_policy,
        declare_subgoal_lookahead,
        declare_pr_gr_tight_coupling,
        declare_depth_cutoff_value,
        declare_dry_run,
        declare_viz_condition,
        declare_write_to_disk,
        place_recognition_node,
        goal_reaching_policy_node,
        visualization_node,
        disk_writer_node,
        launch_ros.actions.SetParameter(name='use_sim_time', value=False),
    ])
