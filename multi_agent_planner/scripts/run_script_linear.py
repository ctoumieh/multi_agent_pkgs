#!/usr/bin/env python3
from math import pi
import subprocess
import time
import os
import csv
import numpy as np

# Package and launch file name
package_name = "multi_agent_planner"
launch_file_name = "agent_planner_long.launch.py"
package_name_env = "env_builder"
launch_file_name_env = "env_builder_dyn_script.launch.py"

# List of expected nodes (that you want to monitor)
expected_nodes = ["/map_builder_node_0", "/agent_node_0", "/env_builder_node"]  # Add the nodes you expect to run
expected_nodes_names = ["map_builder_node_0", "agent_node_0", "env_builder_node"]  # Add the nodes you expect to run

# collision time 
collision_time = 4.7
# create theta and speed list
n_theta = 5 
n_speed = 5 
range_theta = [0*pi/180, 150*pi/180]
range_speed = [2, 5] 
pot_dist_list = [1.2, 2.4, 3.6, 4.8]
traj_keep_list = [6, 5, 4, 3]
# Generate the lists by dividing the range into equal intervals
theta_list = np.linspace(range_theta[0], range_theta[1], n_theta)
speed_list = np.linspace(range_speed[0], range_speed[1], n_speed)
# Number of times to repeat each experiment
n = 5 

# print the list
print(theta_list)
print(speed_list)

def launch_ros2_env(col_time, th, sp):
    """Launch the ROS 2 launch file using package and file name, and return the process object."""
    try:
        # Launch the ROS 2 launch file using package and file name
        process_env = subprocess.Popen(['ros2', 'launch', package_name_env, launch_file_name_env, f'collision_time:={col_time}', f'theta:={th}', f'speed:={sp}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process_env

    except Exception as e:
        print(f"Error launching ROS 2 file: {e}")
        return None


def launch_ros2_agents(pot_dist, traj_keep):
    """Launch the ROS 2 launch file using package and file name, and return the process object."""
    try:
        # Launch the ROS 2 launch file using package and file name
        process = subprocess.Popen(['ros2', 'launch', package_name, launch_file_name, f'potential_dist_max:={pot_dist}', f'traj_keep:={traj_keep}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"Error launching ROS 2 file: {e}")
        return None

def monitor_nodes():
    """Check if all expected nodes are still running."""
    try:
        # Run 'ros2 node list' to get the list of active nodes
        result = subprocess.run(['ros2', 'node', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        running_nodes = result.stdout.decode('utf-8').splitlines()
        # for node in running_nodes:
        #     print(f"Node name: {node}")

        # Check if any expected nodes are missing
        for node in expected_nodes:
            if node not in running_nodes:
                print(f"Error: Node {node} has exited or is not running.")
                return True
        return False
    except Exception as e:
        print(f"Error while monitoring nodes: {e}")
        return True  # In case of any issues, assume there's an error

def kill_ros2_process():
    """Kill the ROS 2 launch file process."""
    try:
        for node in expected_nodes_names:
            print(f"Killing node: {node}")
            subprocess.run(['pkill', '-f', node])
            time.sleep(1)
    except Exception as e:
        print(f"Error killing process: {e}")

# iteration counter
# Main loop for running the process n times
for traj_keep in traj_keep_list:
    for pot_dist in pot_dist_list:
        iter_counter = 0
        for theta in theta_list:
            speed_tally = []
            for speed in speed_list:
                # Initialize tally
                tally = 0
                for i in range(n):
                    print(f"Run {i+1} of {n}")
                    process = launch_ros2_agents(pot_dist, traj_keep)
                    process_env = launch_ros2_env(collision_time, theta, speed)

                    if not process: 
                        print("Failed to launch ROS 2 agents. Skipping iteration.")
                        continue

                    if not process_env: 
                        print("Failed to launch ROS 2 environment. Skipping iteration.")
                        continue

                    # Monitor nodes at intervals (e.g., every 2 seconds)
                    time.sleep(2)
                    error_detected = False
                    time_counter = 0
                    try:
                        while process.poll() is None and time_counter < 15:  # While the process is still running
                            if monitor_nodes():
                                error_detected = True
                                break
                            time_counter = time_counter + 1
                            time.sleep(2)  # Wait 2 seconds before checking again
                    except KeyboardInterrupt:
                        print("Process interrupted manually.")
                        kill_ros2_process()
                        break

                    if error_detected:
                        tally = tally + 1
                        print(f"Error detected in nodes, ending run {i+1}")
                        kill_ros2_process()

                    else:
                        kill_ros2_process()
                        print(f"No error detected in run {i+1}")

                    time.sleep(1)

                speed_tally.append(tally/n)

            filename = f'failure_rate_traj_{traj_keep}_dist_{pot_dist:.1f}.csv'
            if iter_counter == 0:
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(speed_tally)
            else:
                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(speed_tally)

            iter_counter = iter_counter + 1
