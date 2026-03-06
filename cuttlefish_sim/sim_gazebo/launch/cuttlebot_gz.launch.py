# Drop-in replacement for the upstream turtlebot4_gz.launch.py which has a bug:
# it never forwards localization/slam/nav2/use_sim_time to turtlebot4_spawn,
# so AMCL is never launched and LiDAR-based localization silently fails.
#
# Also fixes:
# - map argument not forwarded to localization (launches localization directly)
# - nav2 params not forwarded (launches nav2 directly with custom config)
# - use_sim_time missing on upstream nodes (global SetParameter)
# - world name not forwarded to ros_gz_bridge (upstream spawn never passes it,
#   so LiDAR/camera bridges default to 'warehouse' topic paths)
#
# Usage:
#   ros2 launch <path>/cuttlebot_gz.launch.py \
#       world:=cuttlebot_world localization:=true map:=<path> nav2:=true rviz:=true

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


from launch_ros.actions import Node, PushRosNamespace, SetParameter

# Path to our custom nav2 config with reduced inflation for the divider gap
CUSTOM_NAV2_PARAMS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'nav2.yaml')


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

    # Spawn robot — localization and nav2 are always false here because we
    # launch them ourselves below so we can pass custom config/map arguments.
    # Pass world so it propagates through spawn -> ros_gz_bridge, which uses
    # world-qualified Gazebo topic paths for ALL sensor bridges (LiDAR, camera).
    # Without this, the bridge defaults to 'warehouse' and sensors silently fail.
    # Wrap spawn in a scoped GroupAction so its launch_arguments
    # (localization:=false, nav2:=false) don't leak into the parent context
    # and silently disable our own localization/nav2 actions below.
    spawn = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_gz, 'launch', 'turtlebot4_spawn.launch.py'])),
            launch_arguments=[
                ('namespace', namespace),
                ('rviz', LaunchConfiguration('rviz')),
                ('world', LaunchConfiguration('world')),
                ('model', LaunchConfiguration('model')),
                ('use_sim_time', use_sim_time),
                ('localization', 'false'),
                ('slam', LaunchConfiguration('slam')),
                ('nav2', 'false'),
                ('x', LaunchConfiguration('x')),
                ('y', LaunchConfiguration('y')),
                ('z', LaunchConfiguration('z')),
                ('yaw', LaunchConfiguration('yaw')),
            ]),
    ], scoped=True, forwarding=True)

    # Localization — launch map_server, amcl, and their lifecycle manager directly
    # instead of going through intermediate launch files that can silently fail.
    localization_params = PathJoinSubstitution(
        [pkg_nav, 'config', 'localization.yaml'])
    tf_remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    localization = GroupAction([
        PushRosNamespace(namespace),
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                localization_params,
                {'yaml_filename': LaunchConfiguration('map'),
                 'use_sim_time': use_sim_time},
            ],
            remappings=tf_remappings,
        ),
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                localization_params,
                {'use_sim_time': use_sim_time,
                 'set_initial_pose': True,
                 'initial_pose': {
                     'x': LaunchConfiguration('x'),
                     'y': LaunchConfiguration('y'),
                     'yaw': LaunchConfiguration('yaw'),
                 }},
            ],
            remappings=tf_remappings,
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                {'autostart': True,
                 'use_sim_time': use_sim_time,
                 'node_names': ['map_server', 'amcl']},
            ],
        ),
    ], condition=IfCondition(LaunchConfiguration('localization')))

    # Nav2 — launch with our custom nav2.yaml (reduced inflation_radius for
    # the 1.2m divider gap, increased raytrace range, relaxed progress checker)
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav, 'launch', 'nav2.launch.py'])),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time),
            ('params_file', CUSTOM_NAV2_PARAMS),
        ],
        condition=IfCondition(LaunchConfiguration('nav2')),
    )

    ld = LaunchDescription(ARGUMENTS)
    # Global use_sim_time for ALL nodes (fixes upstream clock_bridge,
    # turtlebot4_node, and hmi_node missing this parameter)
    ld.add_action(SetParameter('use_sim_time', use_sim_time))
    ld.add_action(gazebo)
    ld.add_action(spawn)
    ld.add_action(localization)
    ld.add_action(nav2)
    return ld
