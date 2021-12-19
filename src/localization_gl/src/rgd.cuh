#ifndef _RGD_CUH_
#define _RGD_CUH_

#include "structs.h"
#include "cuda_exception.h"

cudaError_t cudaCalculateParams3D(
		Point *points,
		size_t number_of_points,
		Grid3DParams &in_out_params);

cudaError_t cudaCountOverlaps (
		int threads,
		Point *device_points,
		size_t points_size,
		char* occupied,
		size_t occupied_size,
		Grid3DParams params,
		Particle *device_particles,
		size_t device_particles_size);

#endif
