#ifndef _PF_H_
#define _PF_H_

#include <Eigen/Eigen>
#include "structs.h"

void initialize_host_device_data(HostDeviceData& data);
void compute_occupancy(const std::vector<Point> &host_points,  const Grid3DParams& params, std::vector<char> &host_occupancy_map);
Pose get_pose(const Eigen::Affine3d& m);
Pose updatePose(const Pose& _p, double x, double y, double z, double ox,double oy, double oz);

Eigen::Affine3d get_matrix(const Pose& _p);
void initial_step(HostDeviceData& data);
void particle_filter_step(HostDeviceData& global_structures, const Pose& pose_update, const std::vector<Point>& points_local);
void update_poses(HostDeviceData& data, const Pose& pose_update);
void compute_overlaps(HostDeviceData& data, const std::vector<Point>& points);
void normalize_overlaps(HostDeviceData& data);
void normalize_W(HostDeviceData& data);
void update_propability(HostDeviceData& data);
void resample(HostDeviceData& data);
std::vector<Particle> choose_random_exploration_particles(HostDeviceData& data);
std::vector<Particle> get_motion_model_particles(HostDeviceData& data);


#endif
