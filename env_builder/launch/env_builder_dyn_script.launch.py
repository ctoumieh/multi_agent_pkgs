from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
import math
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    collision_time = float(LaunchConfiguration('collision_time').perform(context))
    theta = float(LaunchConfiguration('theta').perform(context))
    speed = float(LaunchConfiguration('speed').perform(context))

    print("theta: ", theta, "speed: ", speed, "collision_time: ", collision_time) 
    origin_grid = [0.0, -20.0, -8.0]
    dimension_grid = [40.0, 40.0, 20.0]
 
    direction = [-math.cos(theta), -math.sin(theta), 0.0]
    target_pos = [20.18, 0.09, 2.03] 
    x_pos = target_pos[0] - collision_time*direction[0]*speed - origin_grid[0]
    y_pos = target_pos[1] - collision_time*direction[1]*speed - origin_grid[1]
    z_pos = target_pos[2] - collision_time*direction[2]*speed - origin_grid[2]

    position = [x_pos, y_pos, z_pos]

    print("position: ", position, "direction: ", direction) 

    config = os.path.join(
        get_package_share_directory('env_builder'),
        'config',
        # 'env_default_config.yaml'
        # 'env_long_config.yaml'
        # 'env_loop_config.yaml'
        # 'env_new_config.yaml'
        # 'env_crazyflie_config.yaml'
        # 'env_dyn_config.yaml'
        'env_dyn_test_config.yaml'
    )

    params_sub = [{'publish_period': 0.1},
                  {'speed': speed},
                  {'direction': direction},
                  {'position_dyn_obst_vec': position},
                  {'origin_grid': origin_grid},
                  {'dimension_grid': dimension_grid}]
                  
    env_builder_node = Node(
        package='env_builder',
        executable='env_builder_node',
        name='env_builder_node',
        parameters=[config] + params_sub,
        # prefix=['xterm -fa default -fs 10 -e gdb -ex run --args'],
        output='screen',
        emulate_tty=True
    )

    return [env_builder_node]


def generate_launch_description():
    declare_collision_time = DeclareLaunchArgument(
        'collision_time', default_value='4.3',
        description='time at which the obstacle should reach the point [20.18, 0.09, 2.03]')
    declare_theta = DeclareLaunchArgument(
        'theta', default_value='1.57',
        description='angle of collision')
    declare_speed = DeclareLaunchArgument(
        'speed', default_value='1.0',
        description='speed of the obstacle')
    
    return LaunchDescription([
        declare_collision_time,
        declare_theta,
        declare_speed,
        OpaqueFunction(function=launch_setup),  # Use OpaqueFunction for setup
    ])
