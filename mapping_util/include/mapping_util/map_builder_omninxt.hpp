#ifndef MAP_BUILDER_OMNINXT_HPP_
#define MAP_BUILDER_OMNINXT_HPP_

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "env_builder_msgs/msg/voxel_grid_stamped.hpp"
#include "raycast.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.hh"
#include "visualization_msgs/msg/marker_array.hpp"
#include "voxel_grid.hpp"

#include "depth_estimation_ros2/msg/camera_info.hpp"
#include "depth_estimation_ros2/srv/get_camera_info.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

namespace mapping_util {

class MapBuilder : public ::rclcpp::Node {
public:
  MapBuilder();

private:
  void DeclareRosParameters();
  void InitializeRosParameters();
  void PointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void OnShutdown();
  void DisplayCompTime(::std::vector<double> &comp_time);
  void TfCallback(const ::tf2_msgs::msg::TFMessage::SharedPtr msg);
  void PublishFrustum(const ::geometry_msgs::msg::TransformStamped &tf_stamped);
  void EnvironmentVoxelGridCallback(
      const ::env_builder_msgs::msg::VoxelGridStamped::SharedPtr vg_msg);
  ::voxel_grid_util::VoxelGrid
  MergeVoxelGrids(const ::voxel_grid_util::VoxelGrid &vg_old,
                  const ::voxel_grid_util::VoxelGrid &vg_new);
  void RaycastAndClear(::voxel_grid_util::VoxelGrid &vg,
                       const ::Eigen::Vector3d &start,
                       const bool omninxt = false);

  void SetUncertainToUnknown(::voxel_grid_util::VoxelGrid &vg);

  void ClearLine(::voxel_grid_util::VoxelGrid &vg,
                 ::voxel_grid_util::VoxelGrid &vg_final,
                 const ::Eigen::Vector3d &start, const ::Eigen::Vector3d &end,
                 const bool omninxt = false);

  void ClearVoxelsCenter();

  rclcpp::Subscription<env_builder_msgs::msg::VoxelGridStamped>::SharedPtr
      voxel_grid_sub_;
  ::rclcpp::Subscription<::tf2_msgs::msg::TFMessage>::SharedPtr tf_subscriber_;
  ::rclcpp::Publisher<::env_builder_msgs::msg::VoxelGridStamped>::SharedPtr
      voxel_grid_pub_;
  ::rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      frustum_pub_;
  ::rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_sub_;
  rclcpp::Client<depth_estimation_ros2::srv::GetCameraInfo>::SharedPtr
      camera_info_client_;

  ::std::shared_ptr<::tf2_ros::Buffer> tf_buffer_;
  ::std::shared_ptr<::tf2_ros::TransformListener> tf_listener_{nullptr};

  ::std::vector<double> pos_curr_;
  ::Eigen::Matrix3d rot_mat_cam_;
  bool first_transform_received_;
  std::mutex pos_mutex_;

  // Voxel grids
  ::voxel_grid_util::VoxelGrid voxel_grid_curr_;

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
  double fov_x_;
  double fov_y_;
  double fov_y_offset_;
  double frustum_length_;
  bool limited_fov_;
  std::vector<depth_estimation_ros2::msg::CameraInfo> cameras_;
  std::vector<Eigen::Isometry3d> cameras_in_local_grid_;

  // Computation time vectors
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

#endif // MAP_BUILDER_OMNINXT_HPP_
