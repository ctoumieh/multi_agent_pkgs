from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        'n_crossings', default_value='20',
        description='Number of back-and-forth crossings'))

    config_planner = os.path.join(
        get_package_share_directory('multi_agent_planner'),
        'config',
        'agent_omninxt_config.yaml'
    )
    config_mapper = os.path.join(
        get_package_share_directory('mapping_util'),
        'config',
        'map_builder_default_config.yaml'
    )

    # --- Layout parameters (change n_rob here, must be even) ---
    n_rob = 2
    x_left = -2.5
    x_right = 2.5
    z_plane = 1.2
    y_spacing = 1.25
    half_n = n_rob // 2

    use_mapping_util = True
    free_grid = True
    save_stats = True
    voxel_grid_range = [10.0, 10.0, 4.0]
    voxel_size = 0.2
    potential_dist = 0.6
    n_it_decomp = 72 # 42 for 0.3 voxel size
    potential_dist_max = 1.5
    potential_speed_max = 0.2

    # Start positions:
    #   ids 0..half_n-1       → x = x_left
    #   ids half_n..n_rob-1   → x = x_right
    #   y centered around 0, spaced y_spacing apart
    start_positions = []
    for i in range(half_n):
        y = (i - (half_n - 1) / 2.0) * y_spacing
        start_positions.append(
            (x_left, y, z_plane, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    for i in range(half_n):
        y = (i - (half_n - 1) / 2.0) * y_spacing
        start_positions.append(
            (x_right, y, z_plane, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # Initial goals: cross to -x, keep same y and z
    goal_positions = []
    for s in start_positions:
        goal_positions.append((-s[0], s[1], s[2]))

    # --- Mapping nodes ---
    if use_mapping_util:
        for i in range(n_rob):
            node_mapper = Node(
                package='mapping_util',
                executable='map_builder_node',
                name='map_builder_node_{}'.format(i),
                parameters=[config_mapper,
                            {'id': i},
                            {'voxel_grid_range': voxel_grid_range},
                            {'voxel_size': voxel_size},
                            {'free_grid': free_grid},
                            {'potential_dist': potential_dist},
                            {'potential_dist_max': potential_dist_max},
                            {'potential_speed_max': potential_speed_max}],
                output='screen',
                emulate_tty=True,
            )
            ld.add_action(node_mapper)

    # --- Planner nodes ---
    for i in range(n_rob):
        node_planner = Node(
            package='multi_agent_planner',
            executable='agent_node',
            name='agent_node_{}'.format(i),
            parameters=[config_planner,
                        {'state_ini': list(start_positions[i])},
                        {'n_rob': n_rob},
                        {'id': i},
                        {'goal': list(goal_positions[i])},
                        {'use_mapping_util': use_mapping_util},
                        {'voxel_grid_update_period': 10.0},
                        {'voxel_grid_range': voxel_grid_range},
                        {'save_stats': save_stats},
                        {'planning_active': True},
                        {'use_safety_planes': False},
                        {'use_state_estimate': False}],
            output='screen',
            emulate_tty=True,
        )
        ld.add_action(node_planner)

    # --- Goal cycling publisher ---
    goal_pub_node = Node(
        package='multi_agent_planner',
        executable='crossing_goal_publisher.py',
        name='crossing_goal_publisher',
        parameters=[
            {'n_rob': n_rob},
            {'n_crossings': LaunchConfiguration('n_crossings')},
            {'x_left': x_left},
            {'x_right': x_right},
            {'z_plane': z_plane},
            {'y_spacing': y_spacing},
            {'arrival_threshold': 0.5},
            {'check_period': 0.5},
        ],
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(goal_pub_node)

    # --- Collision detector ---
    # Reads drone_radius from config_planner so it stays consistent
    # with the planner's safety radius.
    collision_node = Node(
        package='multi_agent_planner',
        executable='collision_detector.py',
        name='collision_detector',
        parameters=[
            config_planner,
            {'n_rob': n_rob},
            {'check_period': 0.02},
            {'dyn_obs_topic': '/env_builder_node/dyn_obstacles'},
        ],
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(collision_node)

    return ld
