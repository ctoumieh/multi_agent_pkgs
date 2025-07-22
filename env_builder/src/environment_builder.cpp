#include "environment_builder.hpp"

namespace env_builder {
EnvironmentBuilder::EnvironmentBuilder()
    : ::rclcpp::Node("environment_builder") {

  // declare environment parameters
  DeclareRosParameters();

  // initialize parameters
  InitializeRosParameters();

  // create voxel grid
  CreateEmptyVoxelGrid();

  // add static obstacles to the voxel grid
  AddObstacles();

  // save obstacle positions and pointcloud to file if save_obstacles_ is true
  if (save_obstacles_) {
    SaveObstacles();
  }

  // generate the initial positions and sizes of the dynamic obstacles
  GenerateDynObstacles();

  // create static pointcloud from voxel grid
  env_pc_msg_ = CreatePointCloudFromGrid(voxel_grid_shared_ptr_);

  // create pointcloud publisher and publish at constant frequency
  env_pc_pub_ = create_publisher<::sensor_msgs::msg::PointCloud2>(
      "~/" + env_pc_topic_, 1);
  env_dyn_pc_pub_ = create_publisher<::sensor_msgs::msg::PointCloud2>(
      "~/" + env_pc_topic_ + "_dyn", 1);

  // create dynamic obstacles publisher as boxes
  marker_pub_ = this->create_publisher<::visualization_msgs::msg::MarkerArray>("~/dyn_obstacles", 10);

  // create voxel grid publisher and publish at constant frequency
  voxel_grid_pub_ = create_publisher<::env_builder_msgs::msg::VoxelGridStamped>(
      "~/" + env_vg_topic_, 1);
  voxel_grid_timer_ = create_wall_timer(
      ::std::chrono::milliseconds(int(publish_period_ * 1e3)),
      ::std::bind(&EnvironmentBuilder::TimerCallbackEnvironmentVG, this));

  // create service for drones to call and get the local voxel grid
  voxel_grid_service_ = create_service<::env_builder_msgs::srv::GetVoxelGrid>(
      "~/" + get_grid_service_name_,
      ::std::bind(&EnvironmentBuilder::GetVoxelGridService, this,
                  ::std::placeholders::_1, ::std::placeholders::_2));
}

::env_builder_msgs::msg::VoxelGridStamped
EnvironmentBuilder::GenerateVoxelGridMSG(::std::array<double, 3> &position,
                                         ::std::array<double, 3> &range) {

  ::env_builder_msgs::msg::VoxelGridStamped vg_msg;
  vg_msg.voxel_grid.voxel_size = vox_size_;

  // find the origin of the grid
  ::std::array<double, 3> origin;
  origin[0] = (position[0] - range[0] / 2);
  origin[1] = (position[1] - range[1] / 2);
  origin[2] = (position[2] - range[2] / 2);
  origin[0] = floor((origin[0] - origin_grid_[0]) / vox_size_) * vox_size_ +
              origin_grid_[0];
  origin[1] = floor((origin[1] - origin_grid_[1]) / vox_size_) * vox_size_ +
              origin_grid_[1];
  origin[2] = floor((origin[2] - origin_grid_[2]) / vox_size_) * vox_size_ +
              origin_grid_[2];

  vg_msg.voxel_grid.origin = origin;

  // find the range in integer dimensions
  ::std::array<uint32_t, 3> dim;
  dim[0] = floor(range[0] / vox_size_);
  dim[1] = floor(range[1] / vox_size_);
  dim[2] = floor(range[2] / vox_size_);

  vg_msg.voxel_grid.dimension = dim;

  // find the starting index
  ::std::vector<int> start_idx;
  start_idx.push_back(::std::round((origin[0] - origin_grid_[0]) / vox_size_));
  start_idx.push_back(::std::round((origin[1] - origin_grid_[1]) / vox_size_));
  start_idx.push_back(::std::round((origin[2] - origin_grid_[2]) / vox_size_));

  // generate the sub voxel grid from the environment
  vg_msg.voxel_grid.data.resize(dim[0] * dim[1] * dim[2]);
  for (int i = start_idx[0]; i < start_idx[0] + int(dim[0]); i++) {
    for (int j = start_idx[1]; j < start_idx[1] + int(dim[1]); j++) {
      for (int k = start_idx[2]; k < start_idx[2] + int(dim[2]); k++) {
        int i_msg = i - start_idx[0];
        int j_msg = j - start_idx[1];
        int k_msg = k - start_idx[2];
        int idx = i_msg + dim[0] * j_msg + dim[0] * dim[1] * k_msg;
        vg_msg.voxel_grid.data[idx] =
            voxel_grid_shared_ptr_->GetVoxelInt(::Eigen::Vector3i(i, j, k));
        if (free_grid_) {
          if (vg_msg.voxel_grid.data[idx] == -1) {
            vg_msg.voxel_grid.data[idx] = 0;
          }
        }
      }
    }
  }

  vg_msg.header.stamp = now();
  vg_msg.header.frame_id = env_pc_frame_;

  return vg_msg;
}

void EnvironmentBuilder::GetVoxelGridService(
    const ::std::shared_ptr<::env_builder_msgs::srv::GetVoxelGrid::Request>
        request,
    ::std::shared_ptr<::env_builder_msgs::srv::GetVoxelGrid::Response>
        response) {
  // process the request in a new thread to avoid a huge backlog in case
  // multiple agents are requesting information
  ::std::array<double, 3> position = request->position;
  ::std::array<double, 3> range = request->range;

  ::env_builder_msgs::msg::VoxelGridStamped vg_stamped_msg =
      GenerateVoxelGridMSG(position, range);
  response->voxel_grid_stamped = vg_stamped_msg;
}

void EnvironmentBuilder::CreateDynamicPointCloud() {
  ::pcl::PointCloud<::pcl::PointXYZ> cloud_env;

  for (int i = 0; i < pos_vel_dyn_obst_vec_curr_.size(); i++) {
    ::Eigen::Vector3d center_obst = pos_vel_dyn_obst_vec_curr_[i].first;
    ::Eigen::Vector3d dim_obst = size_dyn_obst_ini_vec_[i];

    Eigen::Vector3i start_idx;
    Eigen::Vector3i end_idx;

    for (int i = 0; i < 3; i++) {
      start_idx(i) = std::floor((center_obst(i) - dim_obst(i) / 2) / vox_size_);
      end_idx(i) = std::floor((center_obst(i) + dim_obst(i) / 2) / vox_size_);
    }

    for (int i = start_idx(0); i <= end_idx(0); i++) {
      for (int j = start_idx(1); j <= end_idx(1); j++) {
        for (int k = start_idx(2); k <= end_idx(2); k++) {
          // add obstacles points to point cloud
          ::pcl::PointXYZ pt;
          pt.x = i * vox_size_ + vox_size_ / 2 + origin_grid_[0];
          pt.y = j * vox_size_ + vox_size_ / 2 + origin_grid_[1];
          pt.z = k * vox_size_ + vox_size_ / 2 + origin_grid_[2];
          cloud_env.points.push_back(pt);
        }
      }
    }
  }

  // create pc message
  ::std::shared_ptr<::sensor_msgs::msg::PointCloud2> pc_msg;
  pc_msg = ::std::make_shared<::sensor_msgs::msg::PointCloud2>();
  ::pcl::toROSMsg(cloud_env, *pc_msg);
  pc_msg->header.frame_id = env_pc_frame_;

  env_dyn_pc_msg_ = pc_msg;
}

void EnvironmentBuilder::PublishDynamicBoxes() {

    auto marker_array = visualization_msgs::msg::MarkerArray();

    for (size_t i = 0; i < pos_vel_dyn_obst_vec_curr_.size(); ++i) {
        auto marker = visualization_msgs::msg::Marker();

        // Header
        marker.header.frame_id = env_pc_frame_; 
        marker.header.stamp = this->now();

        // Marker properties
        marker.ns = "boxes";
        marker.id = static_cast<int>(i);
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Position (center)
        marker.pose.position.x = pos_vel_dyn_obst_vec_curr_[i].first.x() + origin_grid_[0];
        marker.pose.position.y = pos_vel_dyn_obst_vec_curr_[i].first.y() + origin_grid_[1];
        marker.pose.position.z = pos_vel_dyn_obst_vec_curr_[i].first.z() + origin_grid_[2];

        // Orientation (no rotation)
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        // Scale (dimensions)
        marker.scale.x = size_dyn_obst_ini_vec_[i].x();
        marker.scale.y = size_dyn_obst_ini_vec_[i].y();
        marker.scale.z = size_dyn_obst_ini_vec_[i].z();

        // Color (green)
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        // Lifetime (0 means forever)
        marker.lifetime = rclcpp::Duration::from_seconds(0);

        marker_array.markers.push_back(marker);
    }

    marker_pub_->publish(marker_array);
}

void EnvironmentBuilder::TimerCallbackEnvironmentVG() {
  UpdateDynObstacles();
  ::env_builder_msgs::msg::VoxelGridStamped voxel_grid_stamped_msg =
      CreateEnvironmentVoxelGrid();
  voxel_grid_pub_->publish(voxel_grid_stamped_msg);

  // publish pointclouds
  CreateDynamicPointCloud();

  env_pc_msg_->header.stamp = now();
  env_pc_pub_->publish(*env_pc_msg_);

  env_dyn_pc_msg_->header.stamp = now();
  env_dyn_pc_pub_->publish(*env_dyn_pc_msg_);

  PublishDynamicBoxes();
}

::env_builder_msgs::msg::VoxelGridStamped
EnvironmentBuilder::CreateEnvironmentVoxelGrid() {

  ::env_builder_msgs::msg::VoxelGridStamped voxel_grid_stamped_msg;

  // set voxel size
  voxel_grid_stamped_msg.voxel_grid.voxel_size = vox_size_;

  // set origin of the grid
  voxel_grid_stamped_msg.voxel_grid.origin[0] = origin_grid_[0];
  voxel_grid_stamped_msg.voxel_grid.origin[1] = origin_grid_[1];
  voxel_grid_stamped_msg.voxel_grid.origin[2] = origin_grid_[2];

  // set the integer dimensions of the grid
  voxel_grid_stamped_msg.voxel_grid.dimension[0] =
      ceil(dimension_grid_[0] / vox_size_);
  voxel_grid_stamped_msg.voxel_grid.dimension[1] =
      ceil(dimension_grid_[1] / vox_size_);
  voxel_grid_stamped_msg.voxel_grid.dimension[2] =
      ceil(dimension_grid_[2] / vox_size_);

  // copy the content of the static obstacles into the final grid that also
  // includes the dynamic obstacles
  if (voxel_grid_shared_ptr_ != nullptr) {
    voxel_grid_dyn_shared_ptr_ =
        std::make_shared<::voxel_grid_util::VoxelGrid>(*voxel_grid_shared_ptr_);
  }

  // add the dynamic obstacles to the dynamic grid
  for (int i = 0; i < pos_vel_dyn_obst_vec_curr_.size(); i++) {
    ::Eigen::Vector3d center_obst = pos_vel_dyn_obst_vec_curr_[i].first;
    ::Eigen::Vector3d vel_obst = pos_vel_dyn_obst_vec_curr_[i].second;
    ::Eigen::Vector3d dim_obst = size_dyn_obst_ini_vec_[i];
    ::voxel_grid_util::AddObstacle(voxel_grid_dyn_shared_ptr_, center_obst,
                                   dim_obst);

    // add the dynamic obstacle to the voxel grid message
    ::env_builder_msgs::msg::Obstacle obst;
    for (int j = 0; j < 3; j++) {
      obst.position[j] = center_obst[j];
      obst.velocity[j] = vel_obst[j];
      obst.dimension[j] = size_dyn_obst_ini_vec_[i][j];
    }
    voxel_grid_stamped_msg.voxel_grid.dyn_obstacles.push_back(obst);
  }

  // set the data of the grid
  for (unsigned int i = 0; i < voxel_grid_dyn_shared_ptr_->GetDataSize(); i++) {
    voxel_grid_stamped_msg.voxel_grid.data.push_back(
        voxel_grid_dyn_shared_ptr_->GetVoxelData(i));
  }

  voxel_grid_stamped_msg.header.stamp = now();
  voxel_grid_stamped_msg.header.frame_id = env_pc_frame_;

  return voxel_grid_stamped_msg;
}

::std::shared_ptr<::sensor_msgs::msg::PointCloud2>
EnvironmentBuilder::CreatePointCloudFromGrid(
    ::voxel_grid_util::VoxelGrid::Ptr voxel_grid_ptr) {
  ::std::shared_ptr<::sensor_msgs::msg::PointCloud2> pc_msg;
  ::Eigen::Vector3d origin_vg = voxel_grid_ptr->GetOrigin();
  ::Eigen::Vector3i dim_vg = voxel_grid_ptr->GetDim();
  double vox_size = voxel_grid_ptr->GetVoxSize();

  ::pcl::PointCloud<::pcl::PointXYZ> cloud_env;

  // add obstacles points to point cloud
  for (int i = 0; i < dim_vg(0); i++) {
    for (int j = 0; j < dim_vg(1); j++) {
      for (int k = 0; k < dim_vg(2); k++) {
        if (voxel_grid_ptr->IsOccupied(Eigen::Vector3i(i, j, k))) {
          ::pcl::PointXYZ pt;
          pt.x = i * vox_size + vox_size / 2 + origin_vg[0];
          pt.y = j * vox_size + vox_size / 2 + origin_vg[1];
          pt.z = k * vox_size + vox_size / 2 + origin_vg[2];
          cloud_env.points.push_back(pt);
        }
      }
    }
  }

  // create pc message
  pc_msg = ::std::make_shared<::sensor_msgs::msg::PointCloud2>();
  ::pcl::toROSMsg(cloud_env, *pc_msg);
  pc_msg->header.frame_id = env_pc_frame_;

  return pc_msg;
}

void EnvironmentBuilder::AddObstacles() {
  srand(rand_seed_);

  int n_obst;
  if (multi_obst_position_) {
    n_obst = position_obst_vec_.size() / 3;
  } else {
    n_obst = n_obst_;
  }

  for (int i = 0; i < n_obst; i++) {

    ::Eigen::Vector3d center_obst;
    if (multi_obst_position_) {
      center_obst = ::Eigen::Vector3d(position_obst_vec_[3 * i],
                                      position_obst_vec_[3 * i + 1],
                                      position_obst_vec_[3 * i + 2]);
    } else {
      // generate the obstacles center we add 0.01 to avoid modulo by 0 which
      // is undefined behavior
      double eps = 0.02;
      center_obst(0) = ((rand() % int((range_obst_[0] + eps) * 100)) / 100) +
                       origin_obst_[0] - origin_grid_[0];
      center_obst(1) = ((rand() % int((range_obst_[1] + eps) * 100)) / 100) +
                       origin_obst_[1] - origin_grid_[1];
      center_obst(2) = ((rand() % int((range_obst_[2] + eps) * 100)) / 100) +
                       origin_obst_[2] - origin_grid_[2];
    }

    ::Eigen::Vector3d dim_obst;
    if (multi_obst_size_) {
      dim_obst =
          ::Eigen::Vector3d(size_obst_vec_[3 * i], size_obst_vec_[3 * i + 1],
                            size_obst_vec_[3 * i + 2]);
    } else {
      // generate the obstacle size
      dim_obst = ::Eigen::Vector3d(size_obst_[0], size_obst_[1], size_obst_[2]);
    }

    ::voxel_grid_util::AddObstacle(voxel_grid_shared_ptr_, center_obst,
                                   dim_obst);
  }
}

void EnvironmentBuilder::GenerateDynObstacles() {
  // update the initial starting position based on whether we have
  srand(rand_seed_ + 1);

  int n_dyn_obst;
  if (multi_dyn_obst_position_) {
    n_dyn_obst = position_dyn_obst_vec_.size() / 3;
  } else {
    n_dyn_obst = n_dyn_obst_;
  }

  for (int i = 0; i < n_dyn_obst; i++) {
    ::Eigen::Vector3d center_dyn_obst;
    if (multi_dyn_obst_position_) {
      center_dyn_obst = ::Eigen::Vector3d(position_dyn_obst_vec_[3 * i],
                                          position_dyn_obst_vec_[3 * i + 1],
                                          position_dyn_obst_vec_[3 * i + 2]);
    } else {
      // generate the obstacles center we add 0.01 to avoid modulo by 0 which
      // is undefined behavior
      double eps = 0.02;
      center_dyn_obst(0) =
          ((rand() % int((range_obst_[0] + eps) * 100)) / 100) +
          origin_obst_[0] - origin_grid_[0];
      center_dyn_obst(1) =
          ((rand() % int((range_obst_[1] + eps) * 100)) / 100) +
          origin_obst_[1] - origin_grid_[1];
      center_dyn_obst(2) =
          ((rand() % int((range_obst_[2] + eps) * 100)) / 100) +
          origin_obst_[2] - origin_grid_[2];
    }

    ::Eigen::Vector3d dim_dyn_obst;
    if (multi_dyn_obst_size_) {
      dim_dyn_obst = ::Eigen::Vector3d(size_dyn_obst_vec_[3 * i],
                                       size_dyn_obst_vec_[3 * i + 1],
                                       size_dyn_obst_vec_[3 * i + 2]);
    } else {
      // generate the obstacle size
      dim_dyn_obst = ::Eigen::Vector3d(size_dyn_obst_[0], size_dyn_obst_[1],
                                       size_dyn_obst_[2]);
    }

    position_dyn_obst_ini_vec_.push_back(center_dyn_obst);
    size_dyn_obst_ini_vec_.push_back(dim_dyn_obst);
  }
}

void EnvironmentBuilder::UpdateDynObstacles() {
  // use the trefoil model
  pos_vel_dyn_obst_vec_curr_.clear();
  int n_dyn_obst = position_dyn_obst_ini_vec_.size();
  for (int i = 0; i < n_dyn_obst; i++) {
    ::std::pair<::Eigen::Vector3d, ::Eigen::Vector3d> pair_pos_vel;
    if (motion_mode_ == 0) {
      ::Eigen::Vector3d direction(direction_[3 * i], direction_[3 * i + 1],
                                  direction_[3 * i + 2]);
      direction.normalize();
      pair_pos_vel =
          LinearMotion(position_dyn_obst_ini_vec_[i], direction, speed_);

    } else if (motion_mode_ == 1) {
      ::Eigen::Vector3d amplitude(amplitude_[3 * i], amplitude_[3 * i + 1],
                                  amplitude_[3 * i + 2]);
      pair_pos_vel =
          TrefoilMotion(position_dyn_obst_ini_vec_[i], amplitude, offset_[i],

                        speed_);
    }
    pos_vel_dyn_obst_vec_curr_.push_back(pair_pos_vel);
  }

  // increment time iteration
  time_iter_++;
}

::std::pair<::Eigen::Vector3d, ::Eigen::Vector3d>
EnvironmentBuilder::LinearMotion(const ::Eigen::Vector3d initial_position,
                                 const ::Eigen::Vector3d direction,
                                 const double speed) {
  ::Eigen::Vector3d pos;
  ::Eigen::Vector3d vel(direction(0) * speed, direction(1) * speed,
                        direction(2) * speed);

  double t = time_iter_ * publish_period_;
  pos(0) = initial_position(0) + vel(0) * t;
  pos(1) = initial_position(1) + vel(1) * t;
  pos(2) = initial_position(2) + vel(2) * t;

  return ::std::make_pair(pos, vel);
}

::std::pair<::Eigen::Vector3d, ::Eigen::Vector3d>
EnvironmentBuilder::TrefoilMotion(const ::Eigen::Vector3d initial_position,
                                  const ::Eigen::Vector3d amplitude,
                                  const double offset, const double speed) {
  ::Eigen::Vector3d pos;
  ::Eigen::Vector3d vel;

  double t = time_iter_ * publish_period_;
  pos(0) = initial_position(0) +
           amplitude(0) *
               (sin(t * speed + offset) + 2 * sin(2 * t * speed + offset));
  pos(1) = initial_position(1) +
           amplitude(1) *
               (cos(t * speed + offset) - 2 * cos(2 * t * speed + offset));
  pos(2) = initial_position(2) + amplitude(2) * (-sin(3 * t * speed + offset));

  vel(0) = amplitude(0) * speed *
           (cos(t * speed + offset) + 4 * cos(2 * t * speed + offset));
  vel(1) = amplitude(1) * speed *
           (-sin(t * speed + offset) + 4 * sin(2 * t * speed + offset));
  vel(2) = amplitude(2) * speed * (-3 * cos(3 * t * speed + offset));

  return ::std::make_pair(pos, vel);
}

void EnvironmentBuilder::SaveObstacles() {
  // save pointcloud to csv file
  ::voxel_grid_util::WriteGridToFile(voxel_grid_shared_ptr_, "map.csv");
}

void EnvironmentBuilder::CreateEmptyVoxelGrid() {
  ::Eigen::Vector3d origin_tmp(origin_grid_.data());
  ::Eigen::Vector3d dimension_tmp(dimension_grid_.data());

  voxel_grid_shared_ptr_ = ::std::make_shared<::voxel_grid_util::VoxelGrid>(
      origin_tmp, dimension_tmp, vox_size_, free_grid_);
}

void EnvironmentBuilder::DeclareRosParameters() {
  declare_parameter("origin_grid", ::std::vector<double>(3, 0.0));
  declare_parameter("dimension_grid", ::std::vector<double>(3, 15.0));
  declare_parameter("vox_size", 0.3);
  declare_parameter("free_grid", true);
  declare_parameter("save_obstacles", false);
  declare_parameter("publish_period", 0.2);

  declare_parameter("range_obst", ::std::vector<double>(3, 12.0));
  declare_parameter("origin_obst", ::std::vector<double>(3, 2.0));
  declare_parameter("rand_seed", 1);

  declare_parameter("multi_obst_size", false);
  declare_parameter("multi_obst_position", false);
  declare_parameter("size_obst", ::std::vector<double>(3, 2.0));
  declare_parameter("size_obst_vec", ::std::vector<double>(3, 0.0));
  declare_parameter("position_obst_vec", ::std::vector<double>(3, 0.0));
  declare_parameter("n_obst", 10);

  declare_parameter("motion_mode", 0);
  declare_parameter("multi_dyn_obst_size", false);
  declare_parameter("multi_dyn_obst_position", false);
  declare_parameter("size_dyn_obst", ::std::vector<double>(3, 2.0));
  declare_parameter("size_dyn_obst_vec", ::std::vector<double>(3, 0.0));
  declare_parameter("position_dyn_obst_vec", ::std::vector<double>(3, 0.0));
  declare_parameter("n_dyn_obst", 0);
  declare_parameter("speed", 1.0);
  declare_parameter("offset", ::std::vector<double>(3, 0.0));
  declare_parameter("amplitude", ::std::vector<double>(3, 0.0));
  declare_parameter("time_iter", 0);
  declare_parameter("direction", ::std::vector<double>(3, 1.0));

  declare_parameter("env_vg_topic", "environment_voxel_grid");
  declare_parameter("env_pc_topic", "environment_pointcloud");
  declare_parameter("env_pc_frame", "map");

  declare_parameter("get_grid_service_name", "get_voxel_grid");
}

void EnvironmentBuilder::InitializeRosParameters() {
  origin_grid_ = get_parameter("origin_grid").as_double_array();
  dimension_grid_ = get_parameter("dimension_grid").as_double_array();
  vox_size_ = get_parameter("vox_size").as_double();
  free_grid_ = get_parameter("free_grid").as_bool();
  save_obstacles_ = get_parameter("save_obstacles").as_bool();
  publish_period_ = get_parameter("publish_period").as_double();

  range_obst_ = get_parameter("range_obst").as_double_array();
  origin_obst_ = get_parameter("origin_obst").as_double_array();
  rand_seed_ = get_parameter("rand_seed").as_int();

  multi_obst_size_ = get_parameter("multi_obst_size").as_bool();
  multi_obst_position_ = get_parameter("multi_obst_position").as_bool();
  size_obst_ = get_parameter("size_obst").as_double_array();
  size_obst_vec_ = get_parameter("size_obst_vec").as_double_array();
  position_obst_vec_ = get_parameter("position_obst_vec").as_double_array();
  n_obst_ = get_parameter("n_obst").as_int();

  motion_mode_ = get_parameter("motion_mode").as_int();
  multi_dyn_obst_size_ = get_parameter("multi_dyn_obst_size").as_bool();
  multi_dyn_obst_position_ = get_parameter("multi_dyn_obst_position").as_bool();
  size_dyn_obst_ = get_parameter("size_dyn_obst").as_double_array();
  size_dyn_obst_vec_ = get_parameter("size_dyn_obst_vec").as_double_array();
  position_dyn_obst_vec_ =
      get_parameter("position_dyn_obst_vec").as_double_array();
  n_dyn_obst_ = get_parameter("n_dyn_obst").as_int();
  speed_ = get_parameter("speed").as_double();
  offset_ = get_parameter("offset").as_double_array();
  amplitude_ = get_parameter("amplitude").as_double_array();
  time_iter_ = get_parameter("time_iter").as_int();
  direction_ = get_parameter("direction").as_double_array();

  env_vg_topic_ = get_parameter("env_vg_topic").as_string();
  env_pc_topic_ = get_parameter("env_pc_topic").as_string();
  env_pc_frame_ = get_parameter("env_pc_frame").as_string();

  get_grid_service_name_ = get_parameter("get_grid_service_name").as_string();
}
} // namespace env_builder
