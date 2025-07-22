from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def generate_launch_description():
    # get config file
    declared_arguments = []
    declared_arguments.append(DeclareLaunchArgument(
        'traj_keep', default_value='6',
        description='trajectory points to keep'))
    declared_arguments.append(DeclareLaunchArgument(
        'potential_dist_max', default_value='3.2',
        description='maximum potential distance'))

    traj_keep = LaunchConfiguration('traj_keep')
    potential_dist_max = LaunchConfiguration('potential_dist_max')
    ld = LaunchDescription(declared_arguments)

    config = os.path.join(
        get_package_share_directory('multi_agent_planner'),
        'config',
        # 'agent_default_config.yaml'
        'agent_agile_config.yaml'
    )

    config_mapper = os.path.join(
        get_package_share_directory('mapping_util'),
        'config',
        'map_builder_default_config.yaml'
    )

    # params
    use_mapping_util = True
    free_grid = True
    voxel_grid_range = [20.0, 20.0, 6.0]
    state_ini = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    goal = [40.0, 0.0, 2.0]
    potential_speed_max = 2.0 
    potential_dist = 1.2
    dmp_search_rad = 4.0
    save_stats = True
    traj_ref_points_to_keep = traj_keep

    if use_mapping_util:
        params_sub = [{'id': 0},
                      {'voxel_grid_range': voxel_grid_range},
                      {'free_grid': free_grid},
                      {'potential_speed_max': potential_speed_max},
                      {'potential_dist': potential_dist},
                      {'potential_dist_max': potential_dist_max}]
        node_mapper = Node(
            package='mapping_util',
            executable='map_builder_node',
            name='map_builder_node_0',
            parameters=[config_mapper] + params_sub,
            # prefix=['xterm -fa default -fs 10 -xrm "XTerm*selectToClipboard: true" -e gdb -ex run --args'],
            # prefix=['valgrind --leak-check=full --track-origins=yes --show-reachable=yes'],
            # prefix=['xterm -fa default -fs 10 -hold -e'],
            output='screen',
            emulate_tty=True,
        )
        ld.add_action(node_mapper)

    # create node
    params_sub = [{'use_mapping_util': use_mapping_util},
                  {'planner_verbose': False},
                  {'voxel_grid_range': voxel_grid_range},
                  {'state_ini': state_ini},
                  {'dmp_search_rad': dmp_search_rad},
                  {'path_vel_min': 5.0},
                  {'path_vel_max': 5.0},
                  {'goal': goal},
                  {'save_stats': save_stats},
                  {'traj_ref_points_to_keep': traj_ref_points_to_keep}]
    agent_node = Node(
        package='multi_agent_planner',
        executable='agent_node',
        name='agent_node_0',
        parameters=[config] + params_sub,
        # prefix=['xterm -fa default -fs 10 -xrm "XTerm*selectToClipboard: true" -e gdb -ex run --args'],
        # prefix=["sudo \"PYTHONPATH=$PYTHONPATH\" \"LD_LIBRARY_PATH=$LD_LIBRARY_PATH\" \"PATH=$PATH\" \"USER=$USER\"  \"GUROBI_HOME=$GUROBI_HOME\" \"GRB_LICENSE_FILE=$GRB_LICENSE_FILE\" -u toumieh bash -c "],
        # prefix=['xterm -fa default -fs 10 -hold -e'],
        # shell=True
        output='screen',
        emulate_tty=True
    )

    ld.add_action(agent_node)

    # create the goal publisher
    # goal_pub_node = Node(package='multi_agent_planner',  # Package name
    #                      executable='goal_pub_node.py',  # Script installed directly as executable
    #                      name='goal_pub_node',  # Node name
    #                      output='screen',  # Print output to the console
    #                      )

    # ld.add_action(goal_pub_node)

    return ld
