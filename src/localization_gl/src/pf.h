#ifndef _PF_H_
#define _PF_H_

#include <Eigen/Eigen>
#include "structs.h"

void initialize_host_device_data(HostDeviceData& data);
void compute_occupancy(std::vector<Point> &host_points, Grid3DParams params, std::vector<char> &host_occupancy_map);
Pose get_pose(Eigen::Affine3d m);
Eigen::Affine3d get_matrix(Pose _p);
void initial_step(HostDeviceData& data);
void particle_filter_step(HostDeviceData& global_structures, const Pose& pose_update, std::vector<Point> points_local);

#endif
