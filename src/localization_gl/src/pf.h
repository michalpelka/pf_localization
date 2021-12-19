#ifndef _PF_H_
#define _PF_H_

#include <Eigen/Eigen>
#include "structs.h"

void initialize_host_device_data(HostDeviceData& data);
void compute_occupancy(std::vector<Point> &host_points, Grid3DParams params, std::vector<char> &host_occupancy_map);
Pose getPose(Eigen::Affine3d m);
Eigen::Affine3d getMatrix(Pose _p);

#endif
