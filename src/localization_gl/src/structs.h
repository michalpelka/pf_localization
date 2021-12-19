#ifndef _STRUCTS_H_
#define _STRUCTS_H_

#include <vector>
#include <cuda_runtime.h>

struct Point{
	float x;
	float y;
	float z;
	char label;
};

enum PointType{
	not_initialised,
	obstacle
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

struct HostDeviceData{
	std::vector<Point> host_map;
	Point *device_map;
	size_t device_map_size;

	std::vector<char> host_occupancy_map;
	char *device_occupancy_map;
	size_t device_occupancy_map_size;

	Grid3DParams map_grid3Dparams;

	char particle_filter_state;

	~HostDeviceData(){
		cudaFree(device_map);
		cudaFree(device_occupancy_map);
	}
};


#endif
