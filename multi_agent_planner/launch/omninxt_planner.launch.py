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
    voxel_grid_range = [20.0, 20.0, 12.0]
    use_mapping_util = True
    free_grid = False
    save_stats = False

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
                {'free_grid': free_grid}
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

    # Add node actions via opaque function
    ld.add_action(OpaqueFunction(function=launch_nodes))

    # to call this function use the syntax:
    # ros2 launch multi_agent_planner swarm_launch.py \
    # n_rob:=5 \
    # id:=0 \
    # start_state:="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" \
    # goal_position:="[5.0, 5.0, 1.0]"

    return ld

