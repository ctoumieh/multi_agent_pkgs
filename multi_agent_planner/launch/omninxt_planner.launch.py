from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def launch_nodes(context, *args, **kwargs):
    import ast

    # Resolve launch arguments
    rob_id = int(LaunchConfiguration('id').perform(context))
    n_rob = int(LaunchConfiguration('n_rob').perform(context))
    start_state_str = LaunchConfiguration('start_state').perform(context)
    goal_position_str = LaunchConfiguration('goal_position').perform(context)

    # Convert to list of floats
    try:
        start_state = list(map(float, ast.literal_eval(start_state_str)))
        goal_position = list(map(float, ast.literal_eval(goal_position_str)))
    except Exception as e:
        raise RuntimeError(f"Failed to parse start_state or goal_position: {e}")

    use_vision_str = LaunchConfiguration('use_vision').perform(context)
    use_vision = use_vision_str.lower() in ['true', '1', 'yes']

    pointcloud_topic = LaunchConfiguration('pointcloud_topic').perform(context) 

    agent_frame = LaunchConfiguration('agent_frame').perform(context)
    swarm_frames_str = LaunchConfiguration('swarm_frames').perform(context)

    try:
        # Convert string representation of list "['a','b']" to python list
        swarm_frames = ast.literal_eval(swarm_frames_str)
    except Exception as e:
        raise RuntimeError(f"Failed to parse swarm_frames: {e}")

    # Config file paths
    config_mapper = os.path.join(
        get_package_share_directory('mapping_util'),
        'config',
        'map_builder_default_config.yaml'
    )
    config_agent = os.path.join(
        get_package_share_directory('multi_agent_planner'),
        'config',
        'agent_omninxt_config.yaml'
    )

    # Optional flags
    voxel_grid_range = [10.0, 10.0, 4.0]
    voxel_size = 0.15
    min_points_per_voxel = 3
    occupancy_threshold = 2
    free_threshold = 2
    inflation_dist = 0.3
    potential_dist = 0.45
    n_it_decomp = 82 # 42 for 0.3 voxel size
    potential_dist_max = 1.2
    potential_speed_max = 2.0
    use_mapping_util = True
    free_grid = False
    save_stats = True
    planner_active = False

    nodes = []

    # Mapping node
    if use_mapping_util:
        node_mapper = Node(
            package='mapping_util',
            executable='map_builder_node',
            name=f'map_builder_node_{rob_id}',
            parameters=[
                config_mapper,
                {'id': rob_id},
                {'voxel_grid_range': voxel_grid_range},
                {'voxel_size': voxel_size},
                {'inflation_dist': inflation_dist},
                {'min_points_per_voxel': min_points_per_voxel},
                {'occupancy_threshold': occupancy_threshold},
                {'free_threshold': free_threshold},
                {'n_it_decomp': n_it_decomp},
                {'potential_dist': potential_dist},
                {'potential_dist_max': potential_dist_max},
                {'potential_speed_max': potential_speed_max},
                {'free_grid': free_grid},
                {'use_vision': use_vision},
                {'pointcloud_topic': pointcloud_topic},
                {'agent_frame': agent_frame},
                {'swarm_frames': swarm_frames},
                {'filter_radius': 0.2},
                {'save_stats': save_stats}
            ],
            # prefix=['xterm -fa default -fs 10 -xrm "XTerm*selectToClipboard: true" -e gdb -ex run --args'],
            # prefix=['xterm -fa default -fs 10 -hold -e'],
            output='screen',
            emulate_tty=True,
        )
        nodes.append(node_mapper)

    # Agent node
    node_agent = Node(
        package='multi_agent_planner',
        executable='agent_node',
        name=f'agent_node_{rob_id}',
        parameters=[
            config_agent,
            {'state_ini': start_state},
            {'n_rob': n_rob},
            {'id': rob_id},
            {'goal': goal_position},
            {'use_mapping_util': use_mapping_util},
            {'planner_active': planner_active},
            {'save_stats': save_stats}
        ],
        # prefix=['xterm -fa default -fs 10 -xrm "XTerm*selectToClipboard: true" -e gdb -ex run --args'],
        # prefix=['xterm -fa default -fs 10 -hold -e'],
        output='screen',
        emulate_tty=True,
    )
    nodes.append(node_agent)

    return nodes

def generate_launch_description():
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument('n_rob', default_value='1'))
    ld.add_action(DeclareLaunchArgument('id', default_value='0'))
    ld.add_action(DeclareLaunchArgument(
        'start_state',
        default_value='[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]'
    ))
    ld.add_action(DeclareLaunchArgument(
        'goal_position',
        default_value='[1.0,1.0,1.0]'
    ))
    ld.add_action(DeclareLaunchArgument(
        'use_vision',
        default_value='False',
        description='Whether to use real vision (DepthEstimation) or simulation'
    ))
    ld.add_action(DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/depth/pointcloud/combined',
        description='Topic name for the input pointcloud when use_vision is True'
    ))
    ld.add_action(DeclareLaunchArgument('agent_frame', default_value='agent_0'))
    ld.add_action(DeclareLaunchArgument('swarm_frames', default_value="['agent_0']"))

    # Add node actions via opaque function
    ld.add_action(OpaqueFunction(function=launch_nodes))

    # to call this function use the syntax:
    # ros2 launch multi_agent_planner swarm_launch.py \
    # n_rob:=5 \
    # id:=0 \
    # start_state:="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" \
    # goal_position:="[5.0, 5.0, 1.0]"

    return ld

