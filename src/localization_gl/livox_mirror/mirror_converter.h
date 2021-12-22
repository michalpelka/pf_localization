#pragma once

#include <livox_ros_driver/CustomMsg.h>
#include <pcl_ros/point_cloud.h>
#include "utils.h"
#include "utils_io.h"
#include <sophus/se3.hpp>

namespace catoptric_livox{
    pcl::PointCloud<pcl::PointXYZI> converterLivoxMirror(const std::vector<catoptric_livox::Mirror>& mirrors, const livox_ros_driver::CustomMsg::ConstPtr& msg);
}
