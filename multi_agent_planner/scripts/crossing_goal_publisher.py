#!/usr/bin/env python3
"""
Publishes crossing goals for the multi-agent planner.

Each drone starts on one side (x_left or x_right) and is sent to the
opposite side.  Once ALL drones arrive within `arrival_threshold` of
their current goal, the goals are flipped and the process repeats for
a total of `n_crossings` crossings.

Drone positions are read from the /agent_{id}/position Marker topic
that the planner already publishes.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker


class CrossingGoalPublisher(Node):
    def __init__(self):
        super().__init__('crossing_goal_publisher')

        # ---------- parameters ----------
        self.declare_parameter('n_rob', 4)
        self.declare_parameter('n_crossings', 3)
        self.declare_parameter('x_left', -2.5)
        self.declare_parameter('x_right', 2.5)
        self.declare_parameter('z_plane', 1.2)
        self.declare_parameter('y_spacing', 1.25)
        self.declare_parameter('arrival_threshold', 0.5)
        self.declare_parameter('check_period', 0.5)

        self.n_rob = self.get_parameter('n_rob').get_parameter_value().integer_value
        self.n_crossings = self.get_parameter('n_crossings').get_parameter_value().integer_value
        self.x_left = self.get_parameter('x_left').get_parameter_value().double_value
        self.x_right = self.get_parameter('x_right').get_parameter_value().double_value
        self.z_plane = self.get_parameter('z_plane').get_parameter_value().double_value
        self.y_spacing = self.get_parameter('y_spacing').get_parameter_value().double_value
        self.arrival_threshold = self.get_parameter('arrival_threshold').get_parameter_value().double_value
        check_period = self.get_parameter('check_period').get_parameter_value().double_value

        half_n = self.n_rob // 2

        # ---------- build home positions (same logic as launch file) ----------
        self.home_positions = []  # (x, y, z) for each drone
        for i in range(half_n):
            y = (i - (half_n - 1) / 2.0) * self.y_spacing
            self.home_positions.append((self.x_left, y, self.z_plane))
        for i in range(half_n):
            y = (i - (half_n - 1) / 2.0) * self.y_spacing
            self.home_positions.append((self.x_right, y, self.z_plane))

        # ---------- state ----------
        # crossing_idx:  0 → first crossing (home→opposite)
        #                1 → second crossing (opposite→home)
        #                ...
        self.crossing_idx = 0
        self.current_goals = [None] * self.n_rob
        self.positions = [None] * self.n_rob  # latest known (x,y,z)

        # The initial goal (crossing 0) was already set by the launch file,
        # so we record it here and start monitoring.
        self._set_goals_for_crossing(0)

        # ---------- publishers & subscribers ----------
        self.goal_pubs = []
        self.pos_subs = []
        for i in range(self.n_rob):
            pub = self.create_publisher(
                PointStamped,
                'agent_{}/goal'.format(i),
                10)
            self.goal_pubs.append(pub)

            sub = self.create_subscription(
                Marker,
                'agent_{}/position'.format(i),
                lambda msg, idx=i: self._position_cb(msg, idx),
                10)
            self.pos_subs.append(sub)

        # ---------- periodic check timer ----------
        self.timer = self.create_timer(check_period, self._check_arrivals)

        self.get_logger().info(
            f'CrossingGoalPublisher started: n_rob={self.n_rob}, '
            f'n_crossings={self.n_crossings}')

    # ------------------------------------------------------------------
    def _goal_for_drone(self, drone_id: int, crossing_idx: int):
        """Return the (x, y, z) goal for a given drone at a given crossing."""
        hx, hy, hz = self.home_positions[drone_id]
        if crossing_idx % 2 == 0:
            # even crossing → go to opposite x
            return (-hx, hy, hz)
        else:
            # odd crossing → go back home
            return (hx, hy, hz)

    def _set_goals_for_crossing(self, crossing_idx: int):
        """Compute and store goals for every drone at this crossing."""
        for i in range(self.n_rob):
            self.current_goals[i] = self._goal_for_drone(i, crossing_idx)

    def _publish_goals(self):
        """Publish the current goals on every agent's goal topic."""
        for i in range(self.n_rob):
            gx, gy, gz = self.current_goals[i]
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.point.x = gx
            msg.point.y = gy
            msg.point.z = gz
            self.goal_pubs[i].publish(msg)

    # ------------------------------------------------------------------
    def _position_cb(self, msg: Marker, idx: int):
        self.positions[idx] = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z)

    # ------------------------------------------------------------------
    def _check_arrivals(self):
        # Don't do anything if we've finished all crossings
        if self.crossing_idx >= self.n_crossings:
            return

        # Check that we have a position for every drone
        if any(p is None for p in self.positions):
            return

        # Check if ALL drones are within threshold of their goal
        all_arrived = True
        for i in range(self.n_rob):
            gx, gy, gz = self.current_goals[i]
            px, py, pz = self.positions[i]
            dist = math.sqrt((px - gx)**2 + (py - gy)**2 + (pz - gz)**2)
            if dist > self.arrival_threshold:
                all_arrived = False
                break

        if all_arrived:
            self.get_logger().info(
                f'All drones arrived — crossing {self.crossing_idx + 1}'
                f'/{self.n_crossings} complete.')
            self.crossing_idx += 1

            if self.crossing_idx < self.n_crossings:
                self._set_goals_for_crossing(self.crossing_idx)
                self._publish_goals()
                direction = 'home' if self.crossing_idx % 2 == 1 else 'opposite'
                self.get_logger().info(
                    f'Sent new goals (crossing {self.crossing_idx + 1}'
                    f'/{self.n_crossings}, heading {direction}).')
            else:
                self.get_logger().info('All crossings finished!')


def main(args=None):
    rclpy.init(args=args)
    node = CrossingGoalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
