#!/usr/bin/env python3
"""
Collision detector for the multi-agent planner simulation.

Subscribes to:
  - /agent_{i}/position           (Marker) for each drone
  - /env_builder_node/dyn_obstacles (MarkerArray) for dynamic obstacles

Logs:
  - drone-drone collisions when distance < 2 * drone_radius
  - drone-obstacle collisions when drone enters the obstacle's AABB
    (inflated by drone_radius)

At shutdown, prints a summary: total collision events, per-pair counts,
minimum inter-drone distance and minimum drone-obstacle distance over
the entire run.

Each unique collision pair is reported only once per "collision event"
(i.e. it must separate beyond the threshold before being reported again),
to avoid flooding the log when two bodies stay close for many frames.
"""

import math
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class CollisionDetector(Node):
    def __init__(self):
        super().__init__('collision_detector')

        # ---------- parameters ----------
        # drone_radius is read from the planner config (agent_*_config.yaml)
        # so it stays consistent with the planner's safety radius.
        self.declare_parameter('n_rob', 2)
        self.declare_parameter('drone_radius', 0.3)
        self.declare_parameter('check_period', 0.02)  # 50 Hz
        self.declare_parameter('dyn_obs_topic',
                               '/env_builder_node/dyn_obstacles')

        self.n_rob = self.get_parameter('n_rob').value
        self.drone_radius = self.get_parameter('drone_radius').value
        check_period = self.get_parameter('check_period').value
        dyn_topic = self.get_parameter('dyn_obs_topic').value

        # ---------- state ----------
        self.positions = [None] * self.n_rob   # latest (x,y,z) per drone
        self.dyn_obstacles = []                # list of (cx,cy,cz, sx,sy,sz)

        # collision bookkeeping
        self.dd_active = set()                 # currently colliding drone-drone pairs
        self.do_active = set()                 # currently colliding (drone, obs) pairs
        self.dd_events = 0
        self.do_events = 0
        self.dd_pair_counts = {}               # (i,j) -> count
        self.do_pair_counts = {}               # (i,obs_idx) -> count
        self.min_dd_dist = float('inf')
        self.min_do_dist = float('inf')

        # ---------- subscribers ----------
        self.pos_subs = []
        for i in range(self.n_rob):
            sub = self.create_subscription(
                Marker,
                f'agent_{i}/position',
                lambda msg, idx=i: self._position_cb(msg, idx),
                10)
            self.pos_subs.append(sub)

        self.dyn_sub = self.create_subscription(
            MarkerArray, dyn_topic, self._dyn_cb, 10)

        # ---------- timer ----------
        self.timer = self.create_timer(check_period, self._check)

        self.get_logger().info(
            f'CollisionDetector started: n_rob={self.n_rob}, '
            f'drone_radius={self.drone_radius}, dyn_topic={dyn_topic}')

    # ------------------------------------------------------------------
    def _position_cb(self, msg: Marker, idx: int):
        self.positions[idx] = (msg.pose.position.x,
                               msg.pose.position.y,
                               msg.pose.position.z)

    def _dyn_cb(self, msg: MarkerArray):
        obs = []
        for m in msg.markers:
            obs.append((
                m.pose.position.x, m.pose.position.y, m.pose.position.z,
                m.scale.x, m.scale.y, m.scale.z,
            ))
        self.dyn_obstacles = obs

    # ------------------------------------------------------------------
    @staticmethod
    def _dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    @staticmethod
    def _point_to_aabb_dist(p, cx, cy, cz, sx, sy, sz):
        """Signed-ish distance from point p to AABB (negative if inside)."""
        dx = max(abs(p[0] - cx) - sx / 2.0, 0.0)
        dy = max(abs(p[1] - cy) - sy / 2.0, 0.0)
        dz = max(abs(p[2] - cz) - sz / 2.0, 0.0)
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    # ------------------------------------------------------------------
    def _check(self):
        # ---- drone-drone ----
        for i in range(self.n_rob):
            if self.positions[i] is None:
                continue
            for j in range(i + 1, self.n_rob):
                if self.positions[j] is None:
                    continue
                d = self._dist(self.positions[i], self.positions[j])
                if d < self.min_dd_dist:
                    self.min_dd_dist = d
                pair = (i, j)
                if d < 2 * self.drone_radius:
                    if pair not in self.dd_active:
                        self.dd_active.add(pair)
                        self.dd_events += 1
                        self.dd_pair_counts[pair] = self.dd_pair_counts.get(pair, 0) + 1
                        self.get_logger().error(
                            f'[COLLISION] drone-drone {i}<->{j}  dist={d:.3f} m')
                else:
                    self.dd_active.discard(pair)

        # ---- drone-obstacle ----
        for i in range(self.n_rob):
            if self.positions[i] is None:
                continue
            for k, obs in enumerate(self.dyn_obstacles):
                cx, cy, cz, sx, sy, sz = obs
                d = self._point_to_aabb_dist(self.positions[i], cx, cy, cz, sx, sy, sz)
                if d < self.min_do_dist:
                    self.min_do_dist = d
                key = (i, k)
                if d < self.drone_radius:
                    if key not in self.do_active:
                        self.do_active.add(key)
                        self.do_events += 1
                        self.do_pair_counts[key] = self.do_pair_counts.get(key, 0) + 1
                        self.get_logger().error(
                            f'[COLLISION] drone-obstacle drone={i} obs={k}  dist={d:.3f} m')
                else:
                    self.do_active.discard(key)

    # ------------------------------------------------------------------
    def print_summary(self):
        self.get_logger().info('===== COLLISION DETECTOR SUMMARY =====')
        self.get_logger().info(
            f'drone-drone collision events:    {self.dd_events}')
        for pair, c in sorted(self.dd_pair_counts.items()):
            self.get_logger().info(f'  {pair[0]}<->{pair[1]}: {c}')
        self.get_logger().info(
            f'drone-obstacle collision events: {self.do_events}')
        for key, c in sorted(self.do_pair_counts.items()):
            self.get_logger().info(f'  drone {key[0]} - obs {key[1]}: {c}')
        if self.min_dd_dist < float('inf'):
            self.get_logger().info(
                f'min inter-drone distance:    {self.min_dd_dist:.3f} m')
        if self.min_do_dist < float('inf'):
            self.get_logger().info(
                f'min drone-obstacle distance: {self.min_do_dist:.3f} m')
        self.get_logger().info('======================================')


def main(args=None):
    rclpy.init(args=args)
    node = CollisionDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
