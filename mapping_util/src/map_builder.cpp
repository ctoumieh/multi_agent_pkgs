#include "map_builder.hpp"

namespace mapping_util {
MapBuilder::MapBuilder() : ::rclcpp::Node("map_builder") {
  DeclareRosParameters();
  InitializeRosParameters();

  on_shutdown(::std::bind(&MapBuilder::OnShutdown, this));

  pos_curr_.resize(3);
  first_transform_received_ = false;

  tf_buffer_ = ::std::make_shared<::tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = ::std::make_shared<::tf2_ros::TransformListener>(*tf_buffer_, this);

  tf_subscriber_ = this->create_subscription<::tf2_msgs::msg::TFMessage>(
      "/tf", 10,
      ::std::bind(&MapBuilder::TfCallback, this, ::std::placeholders::_1));

  ::std::string vg_pub_topic = "agent_" + std::to_string(id_) + "/voxel_grid";
  voxel_grid_pub_ = create_publisher<::env_builder_msgs::msg::VoxelGridStamped>(
      vg_pub_topic, 10);

  ::std::string frustum_pub_topic = "agent_" + std::to_string(id_) + "/fov";
  frustum_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      frustum_pub_topic, 10);

  if (use_vision_) {
      RCLCPP_INFO(this->get_logger(), "Mode: VISION (Real Hardware) - Waiting for DepthEstimation...");

      camera_info_client_ = this->create_client<depth_estimation_ros2::srv::GetCameraInfo>(
          "/depth_estimation_node/get_camera_info");

      while (!camera_info_client_->wait_for_service(std::chrono::seconds(1))) {
        if (!rclcpp::ok()) return;
        RCLCPP_INFO(this->get_logger(), "Vision service not available, waiting...");
      }

      auto request = std::make_shared<depth_estimation_ros2::srv::GetCameraInfo::Request>();
      auto result = camera_info_client_->async_send_request(request);

      if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result) ==
          rclcpp::FutureReturnCode::SUCCESS) {
        cameras_ = result.get()->cameras;
        RCLCPP_INFO(this->get_logger(), "Successfully received camera info.");
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to call service get_camera_info");
      }

      pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
          pointcloud_topic_, 10,
          std::bind(&MapBuilder::PointCloudCallback, this, std::placeholders::_1));

  } else {
      RCLCPP_INFO(this->get_logger(), "Mode: SIMULATION (Using EnvBuilder Grid)");
      voxel_grid_sub_ = create_subscription<::env_builder_msgs::msg::VoxelGridStamped>(
          env_vg_topic_, 10,
          ::std::bind(&MapBuilder::EnvironmentVoxelGridCallback, this,
                      ::std::placeholders::_1));
  }
}

void MapBuilder::DeclareRosParameters() {
  declare_parameter("env_vg_topic", "env_builder_node/environment_voxel_grid");
  declare_parameter("pointcloud_topic", "/depth/pointcloud/combined");
  declare_parameter("id", 0);
  declare_parameter("voxel_grid_range", ::std::vector<double>(3, 10.0));
  declare_parameter("voxel_size", 0.3);
  declare_parameter("world_frame", "world");
  declare_parameter("agent_frame", "agent_0");
  declare_parameter("free_grid", true);
  declare_parameter("min_points_per_voxel", 10);
  declare_parameter("voxel_min_val", -5);
  declare_parameter("voxel_max_val", 7);
  declare_parameter("occupied_threshold", 4);
  declare_parameter("free_threshold", -4);
  declare_parameter("inflation_dist", 0.3);
  declare_parameter("potential_dist", 1.8);
  declare_parameter("potential_dist_max", 3.6);
  declare_parameter("potential_speed_max", 2.0);
  declare_parameter("potential_pow", 4.0);
  declare_parameter("fov_x", M_PI / 2);
  declare_parameter("fov_y", M_PI / 3);
  declare_parameter("fov_y_offset", 0.0);
  declare_parameter("frustum_length", 1.0);
  declare_parameter("limited_fov", true);
  declare_parameter("use_vision", false);
  declare_parameter("save_stats", false); // Default to false
  declare_parameter("swarm_frames", std::vector<std::string>());
  declare_parameter("filter_radius", 0.6);
}

void MapBuilder::InitializeRosParameters() {
  env_vg_topic_ = get_parameter("env_vg_topic").as_string();
  pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
  id_ = get_parameter("id").as_int();
  voxel_grid_range_ = get_parameter("voxel_grid_range").as_double_array();
  voxel_size_ = get_parameter("voxel_size").as_double();
  world_frame_ = get_parameter("world_frame").as_string();

  std::string default_frame = "agent_" + std::to_string(id_);
  if(has_parameter("agent_frame")){
      agent_frame_ = get_parameter("agent_frame").as_string();
  } else {
      agent_frame_ = default_frame;
  }

  free_grid_ = get_parameter("free_grid").as_bool();
  min_points_per_voxel_ = get_parameter("min_points_per_voxel").as_int();
  voxel_min_val_ = get_parameter("voxel_min_val").as_int();
  voxel_max_val_ = get_parameter("voxel_max_val").as_int();
  occupied_threshold_ = get_parameter("occupied_threshold").as_int();
  free_threshold_ = get_parameter("free_threshold").as_int();
  inflation_dist_ = get_parameter("inflation_dist").as_double();
  potential_dist_ = get_parameter("potential_dist").as_double();
  potential_dist_max_ = get_parameter("potential_dist_max").as_double();
  potential_speed_max_ = get_parameter("potential_speed_max").as_double();
  potential_pow_ = get_parameter("potential_pow").as_double();
  fov_x_ = get_parameter("fov_x").as_double();
  fov_y_ = get_parameter("fov_y").as_double();
  fov_y_offset_ = get_parameter("fov_y_offset").as_double();
  frustum_length_ = get_parameter("frustum_length").as_double();
  limited_fov_ = get_parameter("limited_fov").as_bool();
  use_vision_ = get_parameter("use_vision").as_bool();
  save_stats_ = get_parameter("save_stats").as_bool();
  swarm_frames_ = get_parameter("swarm_frames").as_string_array();
  filter_radius_ = get_parameter("filter_radius").as_double();

  swarm_frames_.erase(
    std::remove(swarm_frames_.begin(), swarm_frames_.end(), agent_frame_),
    swarm_frames_.end());
}

// -------------------------------------------------------------------------
// VISION CALLBACK (with comprehensive timing instrumentation)
// -------------------------------------------------------------------------
void MapBuilder::PointCloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  if (!first_transform_received_) return;

  auto t_start_total = std::chrono::high_resolution_clock::now();

  // 1. Get Drone Transform
  geometry_msgs::msg::TransformStamped transform_stamped;
  try {
    transform_stamped = tf_buffer_->lookupTransform(world_frame_, msg->header.frame_id,
                                                    msg->header.stamp);
  } catch (tf2::TransformException &ex) {
    return;
  }

  ::Eigen::Vector3d pos_curr;
  pos_curr[0] = transform_stamped.transform.translation.x;
  pos_curr[1] = transform_stamped.transform.translation.y;
  pos_curr[2] = transform_stamped.transform.translation.z;

  // 2. Initialize Grid (only on first call)
  if (voxel_grid_curr_.GetData().size() == 0) {
    ::Eigen::Vector3d origin;
    origin[0] = (pos_curr[0] - voxel_grid_range_[0] / 2);
    origin[1] = (pos_curr[1] - voxel_grid_range_[1] / 2);
    origin[2] = (pos_curr[2] - voxel_grid_range_[2] / 2);
    for(int i=0; i<3; ++i) origin[i] = floor(origin[i] / voxel_size_) * voxel_size_;

    ::Eigen::Vector3i dim;
    dim[0] = floor(voxel_grid_range_[0] / voxel_size_);
    dim[1] = floor(voxel_grid_range_[1] / voxel_size_);
    dim[2] = floor(voxel_grid_range_[2] / voxel_size_);
    voxel_grid_curr_ = ::voxel_grid_util::VoxelGrid(origin, dim, voxel_size_, false);
    ClearVoxelsCenter();
  }

  // ==================== TIMED SECTION: PCL Transform ====================
  auto t_start_pcl = std::chrono::high_resolution_clock::now();

  // Manual conversion from ROS Msg to Eigen to avoid linker error
  Eigen::Quaterniond q(
      transform_stamped.transform.rotation.w,
      transform_stamped.transform.rotation.x,
      transform_stamped.transform.rotation.y,
      transform_stamped.transform.rotation.z);
  Eigen::Vector3d t(
      transform_stamped.transform.translation.x,
      transform_stamped.transform.translation.y,
      transform_stamped.transform.translation.z);

  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  iso.translate(t);
  iso.rotate(q);
  Eigen::Matrix4f transform_mat = iso.matrix().cast<float>();

  pcl::PointCloud<pcl::PointXYZ> cloud_in;
  pcl::fromROSMsg(*msg, cloud_in);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::transformPointCloud(cloud_in, cloud, transform_mat);

  auto t_end_pcl = std::chrono::high_resolution_clock::now();
  pcl_transform_comp_time_.push_back(
      std::chrono::duration<double, std::milli>(t_end_pcl - t_start_pcl).count());

  // 4. Initialize Temp Grids
  Eigen::Vector3d origin_curr = voxel_grid_curr_.GetOrigin();
  Eigen::Vector3i dim_curr = voxel_grid_curr_.GetDim();
  double vox_size_curr = voxel_grid_curr_.GetVoxSize();

  Eigen::Vector3d origin_obs = origin_curr;
  Eigen::Vector3i dim_obs = dim_curr;
  ::voxel_grid_util::VoxelGrid vg_obstacles(origin_obs, dim_obs, vox_size_curr, true);

  Eigen::Vector3d origin_acc = origin_curr;
  Eigen::Vector3i dim_acc = dim_curr;
  ::voxel_grid_util::VoxelGrid vg_accum(origin_acc, dim_acc, vox_size_curr, true);

  Eigen::Vector3d origin_drn = origin_curr;
  Eigen::Vector3i dim_drn = dim_curr;
  ::voxel_grid_util::VoxelGrid vg_drone(origin_drn, dim_drn, vox_size_curr, true);

  // ==================== TIMED SECTION: Point Counting ====================
  auto t_start_count = std::chrono::high_resolution_clock::now();

  // 5. Swarm Filtering & Counting (OPTIMIZED)
  std::vector<Eigen::Vector3d> other_drones;
  for (const auto& frame : swarm_frames_) {
      try {
          geometry_msgs::msg::TransformStamped t;
          t = tf_buffer_->lookupTransform(world_frame_, frame, msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
          other_drones.emplace_back(t.transform.translation.x, t.transform.translation.y, t.transform.translation.z);
      } catch (...) {}
  }
  const double r_sq = filter_radius_ * filter_radius_;

  // Pre-compute grid parameters to avoid repeated virtual calls
  const Eigen::Vector3d origin = vg_accum.GetOrigin();
  const Eigen::Vector3i dim = vg_accum.GetDim();
  const double vox_size = vg_accum.GetVoxSize();
  const double inv_vox_size = 1.0 / vox_size;

  // Get raw data pointers for direct access (avoid function call overhead)
  std::vector<int8_t>& accum_data = vg_accum.GetData();
  std::vector<int8_t>& drone_data = vg_drone.GetData();

  // Pre-compute bounds for early rejection
  const double x_min = origin[0], x_max = origin[0] + dim[0] * vox_size;
  const double y_min = origin[1], y_max = origin[1] + dim[1] * vox_size;
  const double z_min = origin[2], z_max = origin[2] + dim[2] * vox_size;

  const size_t num_points = cloud.points.size();
  const size_t dim_yz = dim[1] * dim[2];

  // Parallel point counting with thread-local accumulators to avoid atomic ops
  const int num_threads = omp_get_max_threads();
  std::vector<std::vector<int16_t>> local_accum(num_threads, std::vector<int16_t>(accum_data.size(), 0));
  std::vector<std::vector<int16_t>> local_drone(num_threads, std::vector<int16_t>(drone_data.size(), 0));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto& my_accum = local_accum[tid];
    auto& my_drone = local_drone[tid];

    #pragma omp for schedule(static)
    for (size_t p = 0; p < num_points; ++p) {
      const auto& point = cloud.points[p];
      const float px = point.x, py = point.y, pz = point.z;

      // Early bounds check (skip points outside grid)
      if (px < x_min || px >= x_max ||
          py < y_min || py >= y_max ||
          pz < z_min || pz >= z_max) {
        continue;
      }

      // Direct integer coordinate calculation (no Eigen overhead)
      const int i = static_cast<int>((px - origin[0]) * inv_vox_size);
      const int j = static_cast<int>((py - origin[1]) * inv_vox_size);
      const int k = static_cast<int>((pz - origin[2]) * inv_vox_size);

      // Bounds check (should rarely fail after early check, but be safe)
      if (i < 0 || i >= dim[0] || j < 0 || j >= dim[1] || k < 0 || k >= dim[2]) {
        continue;
      }

      // Direct index calculation
      const size_t idx = i * dim_yz + j * dim[2] + k;
      my_accum[idx]++;

      // Check if point belongs to another drone
      bool is_drone = false;
      for (const auto& d_pos : other_drones) {
        const float dx = px - d_pos[0];
        const float dy = py - d_pos[1];
        const float dz = pz - d_pos[2];
        if (dx*dx + dy*dy + dz*dz < r_sq) {
          is_drone = true;
          break;
        }
      }

      if (is_drone) {
        my_drone[idx]++;
      }
    }
  }

  // Merge thread-local results (parallel reduction)
  #pragma omp parallel for schedule(static)
  for (size_t idx = 0; idx < accum_data.size(); ++idx) {
    int16_t sum_accum = 0, sum_drone = 0;
    for (int t = 0; t < num_threads; ++t) {
      sum_accum += local_accum[t][idx];
      sum_drone += local_drone[t][idx];
    }
    // Clamp to int8_t range
    accum_data[idx] = static_cast<int8_t>(std::min<int16_t>(sum_accum, 127));
    drone_data[idx] = static_cast<int8_t>(std::min<int16_t>(sum_drone, 127));
  }

  auto t_end_count = std::chrono::high_resolution_clock::now();
  point_counting_comp_time_.push_back(
      std::chrono::duration<double, std::milli>(t_end_count - t_start_count).count());

  // ==================== TIMED SECTION: Obstacle Map Creation ====================
  auto t_start_obs = std::chrono::high_resolution_clock::now();

  // 6. Create Obstacle Map
  Eigen::Vector3i dim = vg_accum.GetDim();
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
         Eigen::Vector3i coord(i, j, k);
         if (vg_accum.GetVoxelInt(coord) >= min_points_per_voxel_) {
             vg_obstacles.SetVoxelInt(coord, 100);
         } else {
             vg_obstacles.SetVoxelInt(coord, 0);
         }
      }
    }
  }

  auto t_end_obs = std::chrono::high_resolution_clock::now();
  obstacle_map_comp_time_.push_back(
      std::chrono::duration<double, std::milli>(t_end_obs - t_start_obs).count());

  // ==================== TIMED SECTION: Camera Pose Updates ====================
  auto t_start_cam = std::chrono::high_resolution_clock::now();

  // 7. Update Camera Poses
  // Manual conversion
  Eigen::Quaterniond q_drone(
      transform_stamped.transform.rotation.w,
      transform_stamped.transform.rotation.x,
      transform_stamped.transform.rotation.y,
      transform_stamped.transform.rotation.z);
  Eigen::Vector3d t_drone(
      transform_stamped.transform.translation.x,
      transform_stamped.transform.translation.y,
      transform_stamped.transform.translation.z);
  Eigen::Isometry3d drone_pose_in_world = Eigen::Isometry3d::Identity();
  drone_pose_in_world.translate(t_drone);
  drone_pose_in_world.rotate(q_drone);

  cameras_in_local_grid_.clear();
  for (const auto &camera_info : cameras_) {
    // [FIXED] Corrected property names: 'pose.orientation' and 'pose.position'
    Eigen::Quaterniond q_cam(
        camera_info.pose.orientation.w,
        camera_info.pose.orientation.x,
        camera_info.pose.orientation.y,
        camera_info.pose.orientation.z);
    Eigen::Vector3d t_cam(
        camera_info.pose.position.x,
        camera_info.pose.position.y,
        camera_info.pose.position.z);

    Eigen::Isometry3d camera_pose_rel = Eigen::Isometry3d::Identity();
    camera_pose_rel.translate(t_cam);
    camera_pose_rel.rotate(q_cam);

    Eigen::Isometry3d camera_pose_world = drone_pose_in_world * camera_pose_rel;

    Eigen::Vector3d cam_pos_world = camera_pose_world.translation();
    Eigen::Vector3d cam_pos_local = voxel_grid_curr_.GetCoordLocal(cam_pos_world);

    Eigen::Isometry3d cam_pose_final = Eigen::Isometry3d::Identity();
    cam_pose_final.translate(cam_pos_local);
    cam_pose_final.rotate(camera_pose_world.rotation());
    cameras_in_local_grid_.push_back(cam_pose_final);
  }

  auto t_end_cam = std::chrono::high_resolution_clock::now();
  camera_update_comp_time_.push_back(
      std::chrono::duration<double, std::milli>(t_end_cam - t_start_cam).count());

  // ==================== TIMED SECTION: Raycast ====================
  ::Eigen::Vector3d pos_curr_local = voxel_grid_curr_.GetCoordLocal(pos_curr);
  auto t_start_ray = ::std::chrono::high_resolution_clock::now();

  RaycastAndClear(voxel_grid_curr_, vg_obstacles, vg_accum, vg_drone, pos_curr_local);

  auto t_end_ray = ::std::chrono::high_resolution_clock::now();
  raycast_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_ray - t_start_ray).count());

  // ==================== TIMED SECTION: Thresholding ====================
  auto t_start_thresh = std::chrono::high_resolution_clock::now();

  // 9. Thresholding & Create output grid
  Eigen::Vector3d origin_final = voxel_grid_curr_.GetOrigin();
  Eigen::Vector3i dim_final = voxel_grid_curr_.GetDim();
  double vs_final = voxel_grid_curr_.GetVoxSize();
  ::voxel_grid_util::VoxelGrid voxel_grid(origin_final, dim_final, vs_final, true);

  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
        Eigen::Vector3i coord(i, j, k);
        int inter_val = voxel_grid_curr_.GetVoxelInt(coord);
        if (inter_val >= occupied_threshold_) {
          voxel_grid.SetVoxelInt(coord, 100);
        } else if (inter_val <= free_threshold_) {
          voxel_grid.SetVoxelInt(coord, 0);
        } else {
          voxel_grid.SetVoxelInt(coord, -1);
        }
      }
    }
  }

  auto t_end_thresh = std::chrono::high_resolution_clock::now();
  threshold_comp_time_.push_back(
      std::chrono::duration<double, std::milli>(t_end_thresh - t_start_thresh).count());

  // ==================== TIMED SECTION: SetUncertainToUnknown ====================
  auto t_start_unk = ::std::chrono::high_resolution_clock::now();
  SetUncertainToUnknown(voxel_grid);
  auto t_end_unk = ::std::chrono::high_resolution_clock::now();
  uncertain_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_unk - t_start_unk).count());

  // ==================== TIMED SECTION: Inflate ====================
  auto t_start_inf = ::std::chrono::high_resolution_clock::now();
  voxel_grid.InflateObstacles(inflation_dist_);
  auto t_end_inf = ::std::chrono::high_resolution_clock::now();
  inflate_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_inf - t_start_inf).count());

  // ==================== TIMED SECTION: Potential Field ====================
  auto t_start_pot = ::std::chrono::high_resolution_clock::now();
  voxel_grid.CreatePotentialField(potential_dist_, potential_pow_);
  auto t_end_pot = ::std::chrono::high_resolution_clock::now();
  potential_field_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_pot - t_start_pot).count());

  // Publish
  ::env_builder_msgs::msg::VoxelGrid vg_final_msg = ConvertVGUtilToVGMsg(voxel_grid);
  ::env_builder_msgs::msg::VoxelGridStamped vg_final_msg_stamped;
  vg_final_msg_stamped.voxel_grid = vg_final_msg;
  vg_final_msg_stamped.voxel_grid.voxel_size = voxel_size_;
  vg_final_msg_stamped.header.stamp = now();
  vg_final_msg_stamped.header.frame_id = world_frame_;
  voxel_grid_pub_->publish(vg_final_msg_stamped);

  // ==================== TIMED SECTION: Grid Shift ====================
  auto t_start_shift = std::chrono::high_resolution_clock::now();

  // Shift logic
  Eigen::Vector3d origin_shift = voxel_grid_curr_.GetOrigin();
  double vs_shift = voxel_grid_curr_.GetVoxSize();
  Eigen::Vector3i drone_voxel_idx;
  for(int k=0; k<3; k++) drone_voxel_idx[k] = std::floor((pos_curr[k] - origin_shift[k]) / vs_shift);

  Eigen::Vector3i center_voxel_idx = dim / 2;
  Eigen::Vector3i shift = drone_voxel_idx - center_voxel_idx;

  if (shift.norm() > 0) {
    Eigen::Vector3d old_origin = voxel_grid_curr_.GetOrigin();
    Eigen::Vector3d new_origin = old_origin + shift.cast<double>() * voxel_size_;
    ::voxel_grid_util::VoxelGrid shifted_grid(new_origin, dim, voxel_size_, true);

    for (int i = 0; i < dim[0]; i++) {
      for (int j = 0; j < dim[1]; j++) {
        for (int k = 0; k < dim[2]; k++) {
          Eigen::Vector3i new_coord(i, j, k);
          Eigen::Vector3i old_coord = new_coord + shift;
          shifted_grid.SetVoxelInt(new_coord, voxel_grid_curr_.GetVoxelInt(old_coord));
        }
      }
    }
    voxel_grid_curr_ = shifted_grid;
  }

  auto t_end_shift = std::chrono::high_resolution_clock::now();
  shift_comp_time_.push_back(
      std::chrono::duration<double, std::milli>(t_end_shift - t_start_shift).count());

  // Total Time
  auto t_end_total = ::std::chrono::high_resolution_clock::now();
  tot_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_total - t_start_total).count());
}

// -------------------------------------------------------------------------
// SIMULATION CALLBACK
// -------------------------------------------------------------------------
void MapBuilder::EnvironmentVoxelGridCallback(
    const ::env_builder_msgs::msg::VoxelGridStamped::SharedPtr vg_msg) {
  if (first_transform_received_) {
    auto t_start_wall_global = ::std::chrono::high_resolution_clock::now();
    double voxel_size = vg_msg->voxel_grid.voxel_size;

    ::std::array<double, 3> origin_grid = vg_msg->voxel_grid.origin;
    ::Eigen::Vector3d origin;
    pos_mutex_.lock();
    ::Eigen::Vector3d pos_curr(pos_curr_[0], pos_curr_[1], pos_curr_[2]);
    origin[0] = (pos_curr_[0] - voxel_grid_range_[0] / 2);
    origin[1] = (pos_curr_[1] - voxel_grid_range_[1] / 2);
    origin[2] = (pos_curr_[2] - voxel_grid_range_[2] / 2);
    pos_mutex_.unlock();
    for(int i=0; i<3; i++) origin[i] = round((origin[i] - origin_grid[i]) / voxel_size) * voxel_size + origin_grid[i];

    ::Eigen::Vector3i dim;
    dim[0] = floor(voxel_grid_range_[0] / voxel_size);
    dim[1] = floor(voxel_grid_range_[1] / voxel_size);
    dim[2] = floor(voxel_grid_range_[2] / voxel_size);

    ::std::vector<int> start_idx;
    for(int i=0; i<3; i++) start_idx.push_back(::std::round((origin[i] - origin_grid[i]) / voxel_size));

    ::voxel_grid_util::VoxelGrid vg(origin, dim, voxel_size, free_grid_);
    ::std::array<uint32_t, 3> dim_env = vg_msg->voxel_grid.dimension;

    for (int i = start_idx[0]; i < start_idx[0] + int(dim[0]); i++) {
      for (int j = start_idx[1]; j < start_idx[1] + int(dim[1]); j++) {
        for (int k = start_idx[2]; k < start_idx[2] + int(dim[2]); k++) {
          int i_msg = i - start_idx[0]; int j_msg = j - start_idx[1]; int k_msg = k - start_idx[2];
          int idx_env = i + int(dim_env[0]) * j + int(dim_env[0]) * int(dim_env[1]) * k;
          int8_t data_val;
          if (i < 0 || j < 0 || k < 0 || i >= dim_env[0] || j >= dim_env[1] || k >= dim_env[2]) data_val = -1;
          else data_val = vg_msg->voxel_grid.data[idx_env];
          vg.SetVoxelInt(::Eigen::Vector3i(i_msg, j_msg, k_msg), data_val);
          if (free_grid_ && data_val == -1) vg.SetVoxelInt(::Eigen::Vector3i(i_msg, j_msg, k_msg), 0);
          else if (!free_grid_ && data_val == 0) vg.SetVoxelInt(::Eigen::Vector3i(i_msg, j_msg, k_msg), -1);
        }
      }
    }
    if (!free_grid_) {
      if (voxel_grid_curr_.GetData().size() == 0) {
        Eigen::Vector3d o = vg.GetOrigin();
        Eigen::Vector3i d = vg.GetDim();
        double vs = vg.GetVoxSize();
        voxel_grid_curr_ = ::voxel_grid_util::VoxelGrid(o, d, vs, false);
        ClearVoxelsCenter();
      }
      ::Eigen::Vector3d pos_curr_local = vg.GetCoordLocal(pos_curr);
      auto t_start_wall = ::std::chrono::high_resolution_clock::now();
      RaycastAndClear(vg, pos_curr_local);
      auto t_end_wall = ::std::chrono::high_resolution_clock::now();
      raycast_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_wall - t_start_wall).count());

      t_start_wall = ::std::chrono::high_resolution_clock::now();
      voxel_grid_curr_ = MergeVoxelGrids(voxel_grid_curr_, vg);
      t_end_wall = ::std::chrono::high_resolution_clock::now();
      merge_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_wall - t_start_wall).count());
    } else {
      voxel_grid_curr_ = vg;
    }

    ::voxel_grid_util::VoxelGrid voxel_grid = voxel_grid_curr_;

    auto t_start_wall = ::std::chrono::high_resolution_clock::now();
    SetUncertainToUnknown(voxel_grid);
    auto t_end_wall = ::std::chrono::high_resolution_clock::now();
    uncertain_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_wall - t_start_wall).count());

    t_start_wall = ::std::chrono::high_resolution_clock::now();
    voxel_grid.InflateObstacles(inflation_dist_);
    t_end_wall = ::std::chrono::high_resolution_clock::now();
    inflate_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_wall - t_start_wall).count());

    t_start_wall = ::std::chrono::high_resolution_clock::now();
    voxel_grid.CreatePotentialField(potential_dist_, potential_pow_);
    t_end_wall = ::std::chrono::high_resolution_clock::now();
    potential_field_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_wall - t_start_wall).count());

    ::std::vector<Eigen::Vector3d> position_vec, velocity_vec, dimension_vec;
    for (const auto &obs : vg_msg->voxel_grid.dyn_obstacles) {
        position_vec.push_back(Eigen::Vector3d(obs.position[0] + origin_grid[0] - origin[0], obs.position[1] + origin_grid[1] - origin[1], obs.position[2] + origin_grid[2] - origin[2]));
        velocity_vec.push_back(Eigen::Vector3d(obs.velocity[0], obs.velocity[1], obs.velocity[2]));
        dimension_vec.push_back(Eigen::Vector3d(obs.dimension[0], obs.dimension[1], obs.dimension[2]));
    }

    t_start_wall = ::std::chrono::high_resolution_clock::now();
    voxel_grid.CreateDynamicObstaclesPotentialField(position_vec, velocity_vec, dimension_vec, potential_dist_, potential_dist_max_, potential_speed_max_, 50);
    t_end_wall = ::std::chrono::high_resolution_clock::now();
    dyn_obst_field_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_wall - t_start_wall).count());

    ::env_builder_msgs::msg::VoxelGrid vg_final_msg = ConvertVGUtilToVGMsg(voxel_grid);
    ::env_builder_msgs::msg::VoxelGridStamped vg_final_msg_stamped;
    vg_final_msg_stamped.voxel_grid = vg_final_msg;
    vg_final_msg_stamped.voxel_grid.voxel_size = voxel_size;
    vg_final_msg_stamped.header.stamp = now();
    vg_final_msg_stamped.header.frame_id = world_frame_;

    auto t_end_wall_global = ::std::chrono::high_resolution_clock::now();
    tot_comp_time_.push_back(::std::chrono::duration<double, std::milli>(t_end_wall_global - t_start_wall_global).count());

    voxel_grid_pub_->publish(vg_final_msg_stamped);
  }
}

// -------------------------------------------------------------------------
// HELPERS
// -------------------------------------------------------------------------

// Vision Raycaster
void MapBuilder::RaycastAndClear(::voxel_grid_util::VoxelGrid &vg_curr,
                                 const ::voxel_grid_util::VoxelGrid &vg_obstacles,
                                 const ::voxel_grid_util::VoxelGrid &vg_accum,
                                 const ::voxel_grid_util::VoxelGrid &vg_drone,
                                 const ::Eigen::Vector3d &start) {
  ::Eigen::Vector3d origin = vg_curr.GetOrigin();
  ::Eigen::Vector3i dim = vg_curr.GetDim();

  // Border Loops
  ::std::vector<int> k_vec = {0, dim(2) - 1};
  for (int i = 0; i < dim(0); i++) {
    for (int j = 0; j < dim(1); j++) {
      for (int k : k_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg_curr, vg_obstacles, vg_accum, vg_drone, start, end);
      }
    }
  }
  ::std::vector<int> j_vec = {0, dim(1) - 1};
  for (int i = 0; i < dim(0); i++) {
    for (int k = 0; k < dim(2); k++) {
      for (int j : j_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg_curr, vg_obstacles, vg_accum, vg_drone, start, end);
      }
    }
  }
  ::std::vector<int> i_vec = {0, dim(0) - 1};
  for (int j = 0; j < dim(1); j++) {
    for (int k = 0; k < dim(2); k++) {
      for (int i : i_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg_curr, vg_obstacles, vg_accum, vg_drone, start, end);
      }
    }
  }
}

// Sim Raycaster
void MapBuilder::RaycastAndClear(::voxel_grid_util::VoxelGrid &vg,
                                 const ::Eigen::Vector3d &start) {
  ::Eigen::Vector3d origin = vg.GetOrigin();
  ::Eigen::Vector3i dim = vg.GetDim();
  ::voxel_grid_util::VoxelGrid vg_final(origin, dim, vg.GetVoxSize(), false);
  ::std::vector<int> k_vec = {0, dim(2) - 1};
  for (int i = 0; i < dim(0); i++) {
    for (int j = 0; j < dim(1); j++) {
      for (int k : k_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg, vg_final, start, end);
      }
    }
  }
  ::std::vector<int> j_vec = {0, dim(1) - 1};
  for (int i = 0; i < dim(0); i++) {
    for (int k = 0; k < dim(2); k++) {
      for (int j : j_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg, vg_final, start, end);
      }
    }
  }
  ::std::vector<int> i_vec = {0, dim(0) - 1};
  for (int j = 0; j < dim(1); j++) {
    for (int k = 0; k < dim(2); k++) {
      for (int i : i_vec) {
        ::Eigen::Vector3d end(i + 0.5, j + 0.5, k + 0.5);
        ClearLine(vg, vg_final, start, end);
      }
    }
  }
  vg = vg_final;
}

// Vision ClearLine (Smart)
void MapBuilder::ClearLine(::voxel_grid_util::VoxelGrid &vg_curr,
                           const ::voxel_grid_util::VoxelGrid &vg_obstacles,
                           const ::voxel_grid_util::VoxelGrid &vg_accum,
                           const ::voxel_grid_util::VoxelGrid &vg_drone,
                           const ::Eigen::Vector3d &start,
                           const ::Eigen::Vector3d &end) {
  bool in_fov = false;
  ::Eigen::Vector3d start_f = start;

  for (size_t i = 0; i < cameras_in_local_grid_.size(); ++i) {
      const auto &cam_pose = cameras_in_local_grid_[i];
      const auto &cam_info = cameras_[i];
      ::Eigen::Matrix3d rot_mat = cam_pose.rotation();
      ::Eigen::Vector3d x_b = rot_mat.col(0);
      ::Eigen::Vector3d y_b = rot_mat.col(1);
      ::Eigen::Vector3d z_b = rot_mat.col(2);
      ::Eigen::Vector3d cam_position = cam_pose.translation();
      ::Eigen::Vector3d dir = end - cam_position;
      ::Eigen::Vector3d dir_xz = dir - dir.dot(y_b) * y_b; dir_xz.normalize();
      ::Eigen::Vector3d dir_xy = dir - dir.dot(z_b) * z_b; dir_xy.normalize();
      if (dir_xy.dot(x_b) > cos(cam_info.hfov_radians / 2.0) && dir_xz.dot(x_b) > cos(cam_info.vfov_radians / 2.0)) {
        in_fov = true; start_f = cam_position; break;
      }
  }

  if (in_fov) {
    ::Eigen::Vector3d collision_pt;
    ::std::vector<::Eigen::Vector3d> visited_points;
    double max_dist_raycast = (start_f - end).norm();

    // Check against OBSTACLES grid (contains both Walls and Drones)
    bool line_clear = ::path_finding_util::IsLineClear(
      start_f, end, vg_obstacles, max_dist_raycast, collision_pt, visited_points);

    if (!line_clear) {
      // We hit something (Wall or Drone)
      ::Eigen::Vector3d last_point = (end - start_f) * 1e-7 + collision_pt;
      ::Eigen::Vector3i pt_int(last_point[0], last_point[1], last_point[2]);

      // CHECK: Is this a drone?
      int total_count = vg_accum.GetVoxelInt(pt_int);
      int drone_count = vg_drone.GetVoxelInt(pt_int);

      if ((total_count - drone_count) >= min_points_per_voxel_) {
          // It is a WALL (Other points exist) -> Mark Occupied
          int current_val = vg_curr.GetVoxelInt(pt_int);
          vg_curr.SetVoxelInt(pt_int, std::min(voxel_max_val_, current_val + 1));
      } else {
          // It is a DRONE (Mostly drone points) -> Mark Free
          // This "removes" the drone from the map
          int current_val = vg_curr.GetVoxelInt(pt_int);
          vg_curr.SetVoxelInt(pt_int, std::max(voxel_min_val_, current_val - 1));
      }
    }

    // Clear the air up to the collision
    for (size_t i = 0; i < visited_points.size(); i++) {
      Eigen::Vector3i pt(visited_points[i](0), visited_points[i](1), visited_points[i](2));
      int current_val = vg_curr.GetVoxelInt(pt);
      vg_curr.SetVoxelInt(pt, std::max(voxel_min_val_, current_val - 1));
    }
  }
}

// Sim ClearLine
void MapBuilder::ClearLine(::voxel_grid_util::VoxelGrid &vg,
                           ::voxel_grid_util::VoxelGrid &vg_final,
                           const ::Eigen::Vector3d &start,
                           const ::Eigen::Vector3d &end) {
  bool in_fov = false;
  ::Eigen::Vector3d start_f = start;

  if (limited_fov_) {
    ::Eigen::Vector3d x_b = rot_mat_cam_.col(0);
    ::Eigen::Vector3d y_b = rot_mat_cam_.col(1);
    ::Eigen::Vector3d z_b = rot_mat_cam_.col(2);
    ::Eigen::Vector3d dir = end - start;
    ::Eigen::Vector3d dir_xz = dir - dir.dot(y_b) * y_b;
    dir_xz.normalize();
    ::Eigen::Vector3d dir_xy = dir - dir.dot(z_b) * y_b;
    dir_xy.normalize();
    if (dir_xz.dot(x_b) > cos(fov_y_ / 2) &&
        dir_xy.dot(x_b) > cos(fov_x_ / 2)) {
      in_fov = true;
      start_f = start;
    }
  } else {
    in_fov = true;
    start_f = start;
  }

  if (in_fov) {
    ::Eigen::Vector3d collision_pt;
    ::std::vector<::Eigen::Vector3d> visited_points;
    double max_dist_raycast = (start_f - end).norm();
    bool line_clear = ::path_finding_util::IsLineClear(
        start_f, end, vg, max_dist_raycast, collision_pt, visited_points);

    if (line_clear) {
      visited_points.push_back(end);
    } else {
      ::Eigen::Vector3d last_point = (end - start_f) * 1e-7 + collision_pt;
      ::Eigen::Vector3i last_point_int(last_point[0], last_point[1], last_point[2]);
      vg_final.SetVoxelInt(last_point_int, 100);
    }

    for (size_t i = 0; i < visited_points.size() - 1; i++) {
      vg_final.SetVoxelInt(
          ::Eigen::Vector3i(
              (visited_points[i](0) + visited_points[i + 1](0)) / 2.0,
              (visited_points[i](1) + visited_points[i + 1](1)) / 2.0,
              (visited_points[i](2) + visited_points[i + 1](2)) / 2.0),
          0);
    }
  }
}

::voxel_grid_util::VoxelGrid
MapBuilder::MergeVoxelGrids(const ::voxel_grid_util::VoxelGrid &vg_old,
                            const ::voxel_grid_util::VoxelGrid &vg_new) {
  ::voxel_grid_util::VoxelGrid vg_final = vg_new;
  double voxel_size = vg_final.GetVoxSize();
  ::Eigen::Vector3i dim = vg_final.GetDim();
  ::Eigen::Vector3d offset_double = (vg_final.GetOrigin() - vg_old.GetOrigin());
  ::Eigen::Vector3i offset_int;
  offset_int[0] = round(offset_double[0] / voxel_size);
  offset_int[1] = round(offset_double[1] / voxel_size);
  offset_int[2] = round(offset_double[2] / voxel_size);
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      for (int k = 0; k < dim[2]; k++) {
        ::Eigen::Vector3i coord(i, j, k);
        if (vg_final.IsUnknown(coord)) {
          ::Eigen::Vector3i coord_final = coord + offset_int;
          int8_t vox_value = vg_old.GetVoxelInt(coord_final);
          vg_final.SetVoxelInt(coord, vox_value);
        }
      }
    }
  }
  return vg_final;
}

void MapBuilder::SetUncertainToUnknown(::voxel_grid_util::VoxelGrid &vg) {
  ::voxel_grid_util::VoxelGrid vg_final = vg;
  int cube_size = ceil(inflation_dist_ / vg.GetVoxSize());
  ::Eigen::Vector3i dim = vg.GetDim();
  for (int i = cube_size; i < dim[0] - cube_size; i++) {
    for (int j = cube_size; j < dim[1] - cube_size; j++) {
      for (int k = cube_size; k < dim[2] - cube_size; k++) {
        ::Eigen::Vector3i pt(i, j, k);
        if (vg.IsUnknown(pt)) {
          for (int i_new = -cube_size; i_new <= cube_size; i_new++) {
            for (int j_new = -cube_size; j_new <= cube_size; j_new++) {
              for (int k_new = -cube_size; k_new <= cube_size; k_new++) {
                ::Eigen::Vector3i neighbour(i + i_new, j + j_new, k + k_new);
                if (!vg.IsOccupied(neighbour)) {
                  vg_final.SetVoxelInt(neighbour, -1);
                }
              }
            }
          }
        }
      }
    }
  }
  vg = vg_final;
}

void MapBuilder::ClearVoxelsCenter() {
  ::Eigen::Vector3d pos_curr(pos_curr_[0], pos_curr_[1], pos_curr_[2]);
  ::Eigen::Vector3d pos_curr_local = voxel_grid_curr_.GetCoordLocal(pos_curr);
  int i_mid = floor(pos_curr_local(0));
  int j_mid = floor(pos_curr_local(1));
  int k_mid = floor(pos_curr_local(2));
  for (int i = i_mid - 2; i <= i_mid + 2; i++) {
    for (int j = j_mid - 2; j <= j_mid + 2; j++) {
      for (int k = k_mid - 2; k <= k_mid + 2; k++) {
        voxel_grid_curr_.SetVoxelInt(::Eigen::Vector3i(i, j, k), 0);
      }
    }
  }
}

void MapBuilder::TfCallback(const ::tf2_msgs::msg::TFMessage::SharedPtr msg) {
  for (const auto &transform_stamped : msg->transforms) {
    if (transform_stamped.header.frame_id == world_frame_ &&
        transform_stamped.child_frame_id == agent_frame_) {
      const ::geometry_msgs::msg::Transform &transform = transform_stamped.transform;
      pos_curr_[0] = transform.translation.x;
      pos_curr_[1] = transform.translation.y;
      pos_curr_[2] = transform.translation.z;
      ::Eigen::Quaterniond quat_curr;
      quat_curr.x() = transform.rotation.x;
      quat_curr.y() = transform.rotation.y;
      quat_curr.z() = transform.rotation.z;
      quat_curr.w() = transform.rotation.w;
      ::Eigen::Matrix3d rot_mat = quat_curr.normalized().toRotationMatrix();
      ::Eigen::Vector3d x_b = rot_mat.col(0);
      ::Eigen::Vector3d y_b = rot_mat.col(1);
      ::Eigen::Vector3d z_b = rot_mat.col(2);
      ::Eigen::Vector3d x_b_rotated = cos(fov_y_offset_) * x_b + sin(fov_y_offset_) * z_b;
      ::Eigen::Vector3d z_b_rotated = cos(fov_y_offset_) * z_b - sin(fov_y_offset_) * x_b;
      rot_mat_cam_.col(0) = x_b_rotated;
      rot_mat_cam_.col(1) = y_b;
      rot_mat_cam_.col(2) = z_b_rotated;
      first_transform_received_ = true;
      if (limited_fov_ && !use_vision_) PublishFrustum(transform_stamped);
    }
  }
}

void MapBuilder::PublishFrustum(const ::geometry_msgs::msg::TransformStamped &tf_stamped) {
  ::visualization_msgs::msg::MarkerArray marker_array;
  ::geometry_msgs::msg::Point point_0, point_1, point_2, point_3, point_4;
  point_0.x = 0; point_0.y = 0; point_0.z = 0;
  ::Eigen::Vector3d x_b(1, 0, 0); ::Eigen::Vector3d y_b(0, 1, 0); ::Eigen::Vector3d z_b(0, 0, 1);
  ::Eigen::Vector3d x_b_rotated = cos(fov_y_offset_) * x_b + sin(fov_y_offset_) * z_b;
  ::Eigen::Vector3d z_b_rotated = cos(fov_y_offset_) * z_b - sin(fov_y_offset_) * x_b;
  ::Eigen::Vector3d top_left = frustum_length_ * x_b_rotated + tan(fov_x_ / 2) * frustum_length_ * y_b + tan(fov_y_ / 2) * frustum_length_ * z_b;
  ::Eigen::Vector3d bottom_left = frustum_length_ * x_b_rotated + tan(fov_x_ / 2) * frustum_length_ * y_b - tan(fov_y_ / 2) * frustum_length_ * z_b;
  ::Eigen::Vector3d top_right = frustum_length_ * x_b_rotated - tan(fov_x_ / 2) * frustum_length_ * y_b + tan(fov_y_ / 2) * frustum_length_ * z_b;
  ::Eigen::Vector3d bottom_right = frustum_length_ * x_b_rotated - tan(fov_x_ / 2) * frustum_length_ * y_b - tan(fov_y_ / 2) * frustum_length_ * z_b;
  point_1.x = top_left(0); point_1.y = top_left(1); point_1.z = top_left(2);
  point_2.x = bottom_left(0); point_2.y = bottom_left(1); point_2.z = bottom_left(2);
  point_3.x = top_right(0); point_3.y = top_right(1); point_3.z = top_right(2);
  point_4.x = bottom_right(0); point_4.y = bottom_right(1); point_4.z = bottom_right(2);
  ::visualization_msgs::msg::Marker line_0;
  line_0.header.frame_id = agent_frame_;
  line_0.header.stamp = tf_stamped.header.stamp;
  line_0.type = visualization_msgs::msg::Marker::LINE_LIST;
  line_0.action = visualization_msgs::msg::Marker::ADD;
  line_0.scale.x = 0.1; line_0.color.b = 1.0; line_0.color.a = 1.0;
  auto add_line = [&](int id, const auto& p1, const auto& p2) {
      ::visualization_msgs::msg::Marker l = line_0; l.id = id;
      l.points.push_back(p1); l.points.push_back(p2);
      marker_array.markers.push_back(l);
  };
  add_line(0, point_0, point_1); add_line(1, point_0, point_2); add_line(2, point_0, point_3); add_line(3, point_0, point_4);
  add_line(4, point_1, point_2); add_line(5, point_2, point_4); add_line(6, point_4, point_3); add_line(7, point_3, point_1);
  frustum_pub_->publish(marker_array);
}

void MapBuilder::DisplayCompTime(::std::vector<double> &comp_time) {
  if(comp_time.empty()) return;
  double max_t = 0; double min_t = 1e10; double sum_t = 0;
  for (double t : comp_time) { if (t > max_t) max_t = t; if (t < min_t) min_t = t; sum_t += t; }
  double mean_t = sum_t / comp_time.size();
  double std_dev_t = 0;
  for (double t : comp_time) std_dev_t += (t - mean_t) * (t - mean_t);
  std_dev_t = sqrt(std_dev_t / comp_time.size());
  ::std::cout << ::std::endl << "mean: " << mean_t;
  ::std::cout << ::std::endl << "std_dev: " << std_dev_t;
  ::std::cout << ::std::endl << "max: " << max_t;
  ::std::cout << ::std::endl << "min: " << min_t << ::std::endl;
}

void MapBuilder::SaveAndDisplayCompTime(::std::vector<double> &comp_time, ::std::string &filename) {
  if (save_stats_) {
    ::std::string filename_full = "/tmp/planner_logs/" + filename;
    ::std::ofstream myfile;
    myfile.open(filename_full);
    for (int i = 0; i < int(comp_time.size()); i++) {
      myfile << ::std::fixed << comp_time[i] << ",";
    }
    myfile.close();
  }

  // Also print to console
  std::cout << filename << ": ";
  DisplayCompTime(comp_time);
}

void MapBuilder::OnShutdown() {
  ::std::string log_dir = "/tmp/planner_logs/";
  try {
    ::std::filesystem::create_directories(log_dir);
  } catch (const ::std::exception &e) {
    ::std::cerr << "Error creating log directory: " << e.what() << ::std::endl;
  }

  // Save all metrics using the helper function (saves file AND prints stats)
  std::string filename;

  // --- NEW: Vision pipeline detailed timing ---
  filename = "comp_time_pcl_transform_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(pcl_transform_comp_time_, filename);

  filename = "comp_time_point_counting_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(point_counting_comp_time_, filename);

  filename = "comp_time_obstacle_map_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(obstacle_map_comp_time_, filename);

  filename = "comp_time_camera_update_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(camera_update_comp_time_, filename);

  filename = "comp_time_threshold_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(threshold_comp_time_, filename);

  filename = "comp_time_shift_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(shift_comp_time_, filename);

  // --- Existing timing ---
  filename = "comp_time_raycast_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(raycast_comp_time_, filename);

  filename = "comp_time_merge_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(merge_comp_time_, filename);

  filename = "comp_time_uncertain_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(uncertain_comp_time_, filename);

  filename = "comp_time_inflate_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(inflate_comp_time_, filename);

  filename = "comp_time_potential_field_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(potential_field_comp_time_, filename);

  filename = "comp_time_dyn_obst_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(dyn_obst_field_comp_time_, filename);

  filename = "comp_time_total_" + std::to_string(id_) + ".csv";
  SaveAndDisplayCompTime(tot_comp_time_, filename);
}

::voxel_grid_util::VoxelGrid
ConvertVGMsgToVGUtil(::env_builder_msgs::msg::VoxelGrid &vg_msg) {
  ::Eigen::Vector3d origin(vg_msg.origin[0], vg_msg.origin[1], vg_msg.origin[2]);
  ::Eigen::Vector3i dimension(vg_msg.dimension[0], vg_msg.dimension[1], vg_msg.dimension[2]);
  return ::voxel_grid_util::VoxelGrid(origin, dimension, vg_msg.voxel_size, vg_msg.data);
}

::env_builder_msgs::msg::VoxelGrid
ConvertVGUtilToVGMsg(::voxel_grid_util::VoxelGrid &vg) {
  ::env_builder_msgs::msg::VoxelGrid vg_msg;
  ::Eigen::Vector3d origin = vg.GetOrigin();
  vg_msg.origin[0] = origin[0]; vg_msg.origin[1] = origin[1]; vg_msg.origin[2] = origin[2];
  ::Eigen::Vector3i dim = vg.GetDim();
  vg_msg.dimension[0] = dim[0]; vg_msg.dimension[1] = dim[1]; vg_msg.dimension[2] = dim[2];
  vg_msg.voxel_size = vg.GetVoxSize();
  vg_msg.data = vg.GetData();
  return vg_msg;
}
} // namespace mapping_util
