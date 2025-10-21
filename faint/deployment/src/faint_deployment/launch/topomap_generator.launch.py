from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
import rosbag2_py

# Get the python package top directory and make paths relative to it
# Note: This will only work properly if the package was installed as editable (pip install -e .)
from pathlib import Path
import faint
pkg_top_dir = Path(faint.__file__).parent

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def play_bag_with_namespace(context):
    """
    Play a bag file with topics remapped to a namespace
    in order to avoid conflicts with existing topics.
    """

    # Get the bag file and namespace arguments
    bag_file = PathJoinSubstitution([
        LaunchConfiguration('bag_file_dir'),
        LaunchConfiguration('bag_file')
    ]).perform(context)
    namespace = LaunchConfiguration('namespace').perform(context)
    
    # Get the topics in the bag file
    storage_options, converter_options = get_rosbag_options(bag_file)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    topic_names = [topic.name for topic in topic_types]

    # Generate the full command including remap arguments
    cmd = ['ros2', 'bag', 'play', bag_file, '--remap']
    for original_topic in topic_names:
        remapped_topic = f"/{namespace}{original_topic}"
        cmd.append(f"{original_topic}:={remapped_topic}")

    return [ExecuteProcess(cmd=cmd, output='screen')]

def generate_launch_description():
    topomap_generator_node = Node(
        package='faint_deployment',
        executable='topomap_generator_node',
        name='create_topomap',
        output='screen',
        parameters=[{
            'robot_config_path': LaunchConfiguration('robot_config_path'),
            'robot': LaunchConfiguration('robot'),
            'topomap_directory': LaunchConfiguration('topomap_directory'),
            'route_name': LaunchConfiguration('route_name'),
            'dt': LaunchConfiguration('dt'),
            'depth_rgb_sync_slop': LaunchConfiguration('depth_rgb_sync_slop'),
            'depth_rgb_sync_queue_size': LaunchConfiguration('depth_rgb_sync_queue_size'),
            'namespace': LaunchConfiguration('namespace'),
            'camera_type': LaunchConfiguration('camera_type'),
            'obs_mode': LaunchConfiguration('obs_mode'),
        }]
    )

    # Register an event handler to play the bag file after the generator node starts
    start_bag_handler = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=topomap_generator_node,
            on_start=[OpaqueFunction(function=play_bag_with_namespace)]
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_path',
            default_value=pkg_top_dir / 'deployment/src/faint_deployment/config/robots.yaml',
            description='Path to the robot configuration YAML file'
        ),
        DeclareLaunchArgument(
            'bag_file_dir',
            default_value=pkg_top_dir / 'deployment/topomaps/bags',
            description='Directory where the bag files are stored',
        ),
        DeclareLaunchArgument(
            'topomap_directory',
            default_value=pkg_top_dir / 'deployment/topomaps/images',
            description='Directory where the topomap will be stored'
        ),
        DeclareLaunchArgument(
            'robot',
            default_value='turtlebot4',
            description='Name of the robot to use from the configuration file'
        ),
        DeclareLaunchArgument(
            'camera_type',
            default_value='zed2',
            description='Type of camera to use'
        ),
        DeclareLaunchArgument(
            'route_name',
            description='Name of the route for the topomap'
        ),
        DeclareLaunchArgument(
            'dt',
            default_value='0.2',
            description='Time interval in seconds'
        ),
        DeclareLaunchArgument(
            'bag_file',
            description='Name of the bag file to play'
        ),
        DeclareLaunchArgument(
            'depth_rgb_sync_slop',
            default_value='0.1',
            description='Slop time in seconds for synchronizing depth and RGB images'
        ),
        DeclareLaunchArgument(
            'depth_rgb_sync_queue_size',
            default_value='50',
            description='Queue size for synchronizing depth and RGB images'
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='topomap_generator',
            description='Namespace under which to play the bag file and listen to topics'
        ),
        DeclareLaunchArgument(
            'obs_mode',
            default_value='rgb',
            description='What type of images to read from the bag. Options are: rgb, depth, rgbd',
        ),

        # Start the topomap generator node
        topomap_generator_node,

        # Register the event handler
        start_bag_handler

    ])
