# Drop-in replacement for the upstream turtlebot4_gz.launch.py which has a bug:
# it never forwards localization/slam/nav2/use_sim_time to turtlebot4_spawn,
# so AMCL is never launched and LiDAR-based localization silently fails.
#
# Also fixes the map argument not being forwarded to localization by launching
# localization directly instead of through the spawn file.
#
# Usage:
#   ros2 launch <path>/cuttlebot_gz.launch.py \
#       world:=cuttlebot_world localization:=true map:=<path> nav2:=true rviz:=true

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


from launch_ros.actions import Node, PushRosNamespace, SetParameter


ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='false',
                          choices=['true', 'false'],
                          description='Start rviz'),
    DeclareLaunchArgument('world', default_value='warehouse',
                          description='Gazebo world name (warehouse, depot, maze, cuttlebot_world)'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='TurtleBot4 model variant'),
    DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'],
                          description='Use Gazebo simulation clock'),
    DeclareLaunchArgument('localization', default_value='false',
                          choices=['true', 'false'],
                          description='Launch AMCL localization'),
    DeclareLaunchArgument('slam', default_value='false',
                          choices=['true', 'false'],
                          description='Launch SLAM'),
    DeclareLaunchArgument('nav2', default_value='false',
                          choices=['true', 'false'],
                          description='Launch Nav2 navigation stack'),
    DeclareLaunchArgument('map', default_value='',
                          description='Full path to map yaml file for localization'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(
        pose_element, default_value='0.0',
        description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():
    pkg_gz = get_package_share_directory('turtlebot4_gz_bringup')
    pkg_nav = get_package_share_directory('turtlebot4_navigation')

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gz, 'launch', 'sim.launch.py'])),
        launch_arguments=[
            ('world', LaunchConfiguration('world')),
            ('model', LaunchConfiguration('model')),
        ])

    # Spawn robot — localization is always false here because we launch it
    # ourselves below so we can pass the map argument.
    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gz, 'launch', 'turtlebot4_spawn.launch.py'])),
        launch_arguments=[
            ('namespace', namespace),
            ('rviz', LaunchConfiguration('rviz')),
            ('model', LaunchConfiguration('model')),
            ('use_sim_time', use_sim_time),
            ('localization', 'false'),
            ('slam', LaunchConfiguration('slam')),
            ('nav2', LaunchConfiguration('nav2')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw')),
        ])

    # Localization — launch map_server, amcl, and their lifecycle manager directly
    # instead of going through intermediate launch files that can silently fail.
    localization_params = PathJoinSubstitution(
        [pkg_nav, 'config', 'localization.yaml'])
    tf_remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    localization = GroupAction([
        PushRosNamespace(namespace),
        SetParameter('use_sim_time', use_sim_time),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                localization_params,
                {'yaml_filename': LaunchConfiguration('map')},
            ],
            remappings=tf_remappings,
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[localization_params],
            remappings=tf_remappings,
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                {'autostart': True},
                {'node_names': ['map_server', 'amcl']},
            ],
        ),
    ], condition=IfCondition(LaunchConfiguration('localization')))

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gazebo)
    ld.add_action(spawn)
    ld.add_action(localization)
    return ld
