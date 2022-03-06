#ifndef _STRUCTS_H_
#define _STRUCTS_H_

#include <vector>
#include <Eigen/Eigen>
#include <cuda_runtime.h>

#define ROTATION_SE3
//#define ROTATION_TB"crt/math_functions.hpp
struct Point{
	double x;
    double y;
    double z;
	char label;
};

struct PointBucket{
    Eigen::Vector3f p;
    uint32_t index_of_bucket;
};

enum PointType{
	not_initialised,
	obstacle,
	ground
};

struct Grid3DParams{
	float bounding_box_min_X;
	float bounding_box_min_Y;
	float bounding_box_min_Z;
	float bounding_box_max_X;
	float bounding_box_max_Y;
	float bounding_box_max_Z;
	float bounding_box_extension;
	int number_of_buckets_X;
	int number_of_buckets_Y;
	int number_of_buckets_Z;
	long long unsigned int number_of_buckets;
	float resolution_X;
	float resolution_Y;
	float resolution_Z;
};

enum ParticleFilterState{
	initial,
	normal
};

enum ParticleStatus{
	to_kill,
	to_alive
};

struct Position
{
	double x;
	double y;
	double z;
};

struct Orientation
{
	double x_angle_rad;
	double y_angle_rad;
	double z_angle_rad;
};

struct Pose
{
#ifdef ROTATION_TB
	Position p;
	Orientation o;
	Pose(){p.x = p.y = p.z = o.x_angle_rad = o.y_angle_rad = o.z_angle_rad = 0.0;};
#endif
#ifdef ROTATION_SE3
    Eigen::Vector3d p;
    Eigen::Matrix3d o;
    Pose():p{0,0,0},o{Eigen::Matrix3d::Identity()}{}
#endif

};

struct Particle{
	double W;
    double nW;
    double overlap;
	Pose pose;
	char status;
	bool is_tracking;
};

struct HostDeviceData{
	std::vector<Point> host_map;
    std::vector<Point> host_travesability;

    Point *device_map;
	size_t device_map_size;

	std::vector<char> host_occupancy_map;
	char *device_occupancy_map;
	size_t device_occupancy_map_size;

	Grid3DParams map_grid3Dparams;

	char particle_filter_state;
	std::vector<Particle> particle_filter_initial_guesses;
	std::vector<Particle> particles;

	//float initial_w;
	float initial_w_exploration_particles;
	int max_particles;

	std::array<double,6> std_update;

	float min_dump_propability_no_observations;
	float min_dump_propability_tracking;
	float min_dump_propability;

	float percent_particles_from_initial;
	int number_of_replicated_best_particles_motion_model;
	int number_of_replicatations_motion_model;

	float std_motion_model_x;
	float std_motion_model_y;
	float std_motion_model_z_angle_deg;


	~HostDeviceData(){
		cudaFree(device_map);
		cudaFree(device_occupancy_map);
	}
	float step_time;
	int resampling_scheme=0;
};

struct GridParameters2D_XY {
    float bounding_box_min_X;
    float bounding_box_min_Y;
    float bounding_box_max_X;
    float bounding_box_max_Y;
    float bounding_box_extension;
    int number_of_buckets_X;
    int number_of_buckets_Y;
    long long unsigned int number_of_buckets;
    float resolution_X;
    float resolution_Y;
};


#endif
