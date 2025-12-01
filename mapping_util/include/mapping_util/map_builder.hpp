#ifndef MAPPING_UTIL_MAP_BUILDER_CLASS_H_
#define MAPPING_UTIL_MAP_BUILDER_CLASS_H_

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "path_tools.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"
#include "voxel_grid.hpp"
#include <env_builder_msgs/msg/voxel_grid_stamped.hpp>
#include <env_builder_msgs/srv/get_voxel_grid.hpp>
#include <iostream>
#include <mutex>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <stdlib.h>
#include <string>
#include <vector>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Include the service for real vision
#include "depth_estimation_ros2/srv/get_camera_info.hpp"
#include "depth_estimation_ros2/msg/camera_info.hpp"

namespace mapping_util {
class MapBuilder : public ::rclcpp::Node {
public:
  MapBuilder();

private:
  /*-------------- methods ---------------*/
  // declare ros parameters
  void DeclareRosParameters();

  // initialize ros parameters
  void InitializeRosParameters();

  // callback for when we receive the map from the environment (SIMULATION)
  void EnvironmentVoxelGridCallback(
      const ::env_builder_msgs::msg::VoxelGridStamped::SharedPtr vg_msg);

  // callback for when we receive pointclouds (VISION)
  void PointCloudCallback(
      const ::sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // merge 2 grids together
  ::voxel_grid_util::VoxelGrid
  MergeVoxelGrids(const ::voxel_grid_util::VoxelGrid &vg_old,
                  const ::voxel_grid_util::VoxelGrid &vg_new);

  // Sim Raycaster
  void RaycastAndClear(::voxel_grid_util::VoxelGrid &vg,
                       const ::Eigen::Vector3d &start);

  // Vision Raycaster (The smart one)
  void RaycastAndClear(::voxel_grid_util::VoxelGrid &vg_curr,
                       const ::voxel_grid_util::VoxelGrid &vg_obstacles,
                       const ::voxel_grid_util::VoxelGrid &vg_accum,
                       const ::voxel_grid_util::VoxelGrid &vg_drone,
                       const ::Eigen::Vector3d &start);

  void SetUncertainToUnknown(::voxel_grid_util::VoxelGrid &vg);

  // Sim ClearLine
  void ClearLine(::voxel_grid_util::VoxelGrid &vg,
                 ::voxel_grid_util::VoxelGrid &vg_final,
                 const ::Eigen::Vector3d &start, const ::Eigen::Vector3d &end);

  // Vision ClearLine
  void ClearLine(::voxel_grid_util::VoxelGrid &vg_curr,
                 const ::voxel_grid_util::VoxelGrid &vg_obstacles,
                 const ::voxel_grid_util::VoxelGrid &vg_accum,
                 const ::voxel_grid_util::VoxelGrid &vg_drone,
                 const ::Eigen::Vector3d &start, const ::Eigen::Vector3d &end);

  void ClearVoxelsCenter();
  void TfCallback(const ::tf2_msgs::msg::TFMessage::SharedPtr msg);
  void DisplayCompTime(::std::vector<double> &comp_time);
  void PublishFrustum(const ::geometry_msgs::msg::TransformStamped &tf_stamped);
  void OnShutdown();

  /*-------------- member variables ---------------*/
  // ROS parameters
  ::std::string env_vg_topic_;
  ::std::string pointcloud_topic_;
  int id_;
  ::std::vector<double> voxel_grid_range_;
  double voxel_size_;
  ::std::string world_frame_;
  ::std::string agent_frame_;
  bool free_grid_;
  int min_points_per_voxel_;
  int voxel_min_val_;
  int voxel_max_val_;
  int occupied_threshold_;
  int free_threshold_;
  double inflation_dist_;
  double potential_dist_;
  double potential_dist_max_;
  double potential_speed_max_;
  double potential_pow_;
  double fov_x_, fov_y_, fov_y_offset_, frustum_length_;
  bool limited_fov_;

  // New Parameters
  bool use_vision_;
  std::vector<std::string> swarm_frames_;
  double filter_radius_;

  // Publishers/Subscribers
  ::rclcpp::Subscription<::env_builder_msgs::msg::VoxelGridStamped>::SharedPtr voxel_grid_sub_;
  ::rclcpp::Subscription<::sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

  // Service Client
  ::rclcpp::Client<::depth_estimation_ros2::srv::GetCameraInfo>::SharedPtr camera_info_client_;

  ::std::shared_ptr<::tf2_ros::Buffer> tf_buffer_;
  ::std::shared_ptr<::tf2_ros::TransformListener> tf_listener_;
  ::rclcpp::Subscription<::tf2_msgs::msg::TFMessage>::SharedPtr tf_subscriber_;
  ::rclcpp::Publisher<::env_builder_msgs::msg::VoxelGridStamped>::SharedPtr voxel_grid_pub_;
  ::rclcpp::Publisher<::visualization_msgs::msg::MarkerArray>::SharedPtr frustum_pub_;

  // State variables
  ::std::vector<double> pos_curr_;
  ::Eigen::Matrix3d rot_mat_cam_;
  bool first_transform_received_;
  ::voxel_grid_util::VoxelGrid voxel_grid_curr_;
  ::std::mutex pos_mutex_;

  // Camera Info Storage
  std::vector<depth_estimation_ros2::msg::CameraInfo> cameras_;
  std::vector<Eigen::Isometry3d> cameras_in_local_grid_;

  // Timing
  ::std::vector<double> raycast_comp_time_;
  ::std::vector<double> merge_comp_time_;
  ::std::vector<double> uncertain_comp_time_;
  ::std::vector<double> inflate_comp_time_;
  ::std::vector<double> potential_field_comp_time_;
  ::std::vector<double> dyn_obst_field_comp_time_;
  ::std::vector<double> tot_comp_time_;
};

::voxel_grid_util::VoxelGrid
ConvertVGMsgToVGUtil(::env_builder_msgs::msg::VoxelGrid &vg_msg);

::env_builder_msgs::msg::VoxelGrid
ConvertVGUtilToVGMsg(::voxel_grid_util::VoxelGrid &vg);
} // namespace mapping_util
#endif
