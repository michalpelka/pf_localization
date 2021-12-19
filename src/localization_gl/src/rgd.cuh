#ifndef _RGD_CUH_
#define _RGD_CUH_

#include "structs.h"
#include "cuda_exception.h"

cudaError_t cudaCalculateParams3D(
		Point *points,
		size_t number_of_points,
		Grid3DParams &in_out_params);

#endif
