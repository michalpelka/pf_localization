#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <cuda_runtime.h>

#include "rgd.cuh"


cudaError_t cudaCalculateParams3D(
		Point *points,
		size_t number_of_points,
		Grid3DParams &in_out_params)
{
	cudaError_t errCUDA = ::cudaSuccess;

	try
		{
		thrust::device_ptr<Point> t_cloud(points);

		thrust::pair<thrust::device_ptr<Point>,thrust::device_ptr<Point> >
		minmaxX=thrust::minmax_element(t_cloud,t_cloud + number_of_points,
				[] __host__ __device__(const Point & a, const Point & b) { return (a.x < b.x) ;});


		thrust::pair<thrust::device_ptr<Point>,thrust::device_ptr<Point> >
		minmaxY=thrust::minmax_element(t_cloud,t_cloud + number_of_points,
				[] __host__ __device__(const Point & a, const Point & b) { return (a.y < b.y) ;});

		thrust::pair<thrust::device_ptr<Point>,thrust::device_ptr<Point> >
		minmaxZ=thrust::minmax_element(t_cloud,t_cloud + number_of_points,
				[] __host__ __device__(const Point & a, const Point & b) { return (a.z < b.z) ;});


		Point minX, maxX, minY, maxY, minZ, maxZ;

		errCUDA = cudaMemcpy(&minX,minmaxX.first.get(),sizeof(Point),cudaMemcpyDeviceToHost);
		if(errCUDA != ::cudaSuccess)return errCUDA;
		errCUDA = cudaMemcpy(&maxX,minmaxX.second.get(),sizeof(Point),cudaMemcpyDeviceToHost);
		if(errCUDA != ::cudaSuccess)return errCUDA;

		errCUDA = cudaMemcpy(&minY,minmaxY.first.get(),sizeof(Point),cudaMemcpyDeviceToHost);
		if(errCUDA != ::cudaSuccess)return errCUDA;
		errCUDA = cudaMemcpy(&maxY,minmaxY.second.get(),sizeof(Point),cudaMemcpyDeviceToHost);
		if(errCUDA != ::cudaSuccess)return errCUDA;

		errCUDA = cudaMemcpy(&minZ,minmaxZ.first.get(),sizeof(Point),cudaMemcpyDeviceToHost);
		if(errCUDA != ::cudaSuccess)return errCUDA;
		errCUDA = cudaMemcpy(&maxZ,minmaxZ.second.get(),sizeof(Point),cudaMemcpyDeviceToHost);
		if(errCUDA != ::cudaSuccess)return errCUDA;


		maxX.x += in_out_params.bounding_box_extension;
		minX.x -= in_out_params.bounding_box_extension;

		maxY.y += in_out_params.bounding_box_extension;
		minY.y -= in_out_params.bounding_box_extension;

		maxZ.z += in_out_params.bounding_box_extension;
		minZ.z -= in_out_params.bounding_box_extension;


		std::cout << "minX.x: " << minX.x << " maxX.x: " << maxX.x << std::endl;
		std::cout << "minY.y: " << minY.y << " maxY.y: " << maxY.y << std::endl;
		std::cout << "minZ.z: " << minZ.z << " maxZ.z: " << maxZ.z << std::endl;


		long long unsigned int number_of_buckets_X=((maxX.x-minX.x)/in_out_params.resolution_X)+1;
		long long unsigned int number_of_buckets_Y=((maxY.y-minY.y)/in_out_params.resolution_Y)+1;
		long long unsigned int number_of_buckets_Z=((maxZ.z-minZ.z)/in_out_params.resolution_Z)+1;



		in_out_params.number_of_buckets_X = number_of_buckets_X;
		in_out_params.number_of_buckets_Y = number_of_buckets_Y;
		in_out_params.number_of_buckets_Z = number_of_buckets_Z;



		in_out_params.number_of_buckets   = static_cast<long long unsigned int>(number_of_buckets_X) *
			static_cast<long long unsigned int>(number_of_buckets_Y) * static_cast<long long unsigned int>(number_of_buckets_Z);

		in_out_params.bounding_box_max_X = maxX.x;
		in_out_params.bounding_box_min_X = minX.x;
		in_out_params.bounding_box_max_Y = maxY.y;
		in_out_params.bounding_box_min_Y = minY.y;
		in_out_params.bounding_box_max_Z = maxZ.z;
		in_out_params.bounding_box_min_Z = minZ.z;

		std::cout << "number_of_buckets: " << in_out_params.number_of_buckets << std::endl;
		}
			catch(thrust::system_error &e)
			{
				return cudaGetLastError();
			}
			catch(std::bad_alloc &e)
			{
				return cudaGetLastError();
			}
	return cudaGetLastError();
}

__global__ void  kernel_cudaCountOverlaps (
		Point *device_points,
		size_t points_size,
		char* occupied,
		size_t occupied_size,
		Grid3DParams params,
		Particle *device_particles,
		size_t device_particles_size)
{
	size_t index_of_particle = blockIdx.x*blockDim.x + threadIdx.x;

	if (index_of_particle < device_particles_size){
		device_particles[index_of_particle].status = ParticleStatus::to_alive;
		device_particles[index_of_particle].overlap = 0;
		Pose pose = device_particles[index_of_particle].pose;
		double sx = sin(pose.o.x_angle_rad);
		double cx = cos(pose.o.x_angle_rad);
		double sy = sin(pose.o.y_angle_rad);
		double cy = cos(pose.o.y_angle_rad);
		double sz = sin(pose.o.z_angle_rad);
		double cz = cos(pose.o.z_angle_rad);
		float sum_good_hits = 0;

		for(size_t i = 0 ; i < points_size; i++){
			Point pSourceLocal = device_points[i];
			Point pSourceGlobal = pSourceLocal;
			pSourceGlobal.x = (cy * cz) * pSourceLocal.x + (-cy * sz) * pSourceLocal.y + (sy) * pSourceLocal.z + pose.p.x;
			pSourceGlobal.y = (cz * sx * sy + cx * sz) * pSourceLocal.x + (cx * cz - sx * sy * sz) * pSourceLocal.y + (-cy * sx) * pSourceLocal.z + pose.p.y;
			pSourceGlobal.z = (-cx * cz * sy + sx * sz) * pSourceLocal.x + (cz * sx + cx * sy * sz) * pSourceLocal.y + (cx * cy) * pSourceLocal.z + pose.p.z;


			if(pSourceGlobal.x < params.bounding_box_min_X || pSourceGlobal.x > params.bounding_box_max_X)continue;
			if(pSourceGlobal.y < params.bounding_box_min_Y || pSourceGlobal.y > params.bounding_box_max_Y)continue;
			if(pSourceGlobal.z < params.bounding_box_min_Z || pSourceGlobal.z > params.bounding_box_max_Z)continue;

			long long unsigned int ix = (pSourceGlobal.x - params.bounding_box_min_X) / params.resolution_X;
			long long unsigned int iy = (pSourceGlobal.y - params.bounding_box_min_Y) / params.resolution_Y;
			long long unsigned int iz = (pSourceGlobal.z - params.bounding_box_min_Z) / params.resolution_Z;

			long long unsigned int index_bucket = ix* static_cast<long long unsigned int>(params.number_of_buckets_Y) *
						static_cast<long long unsigned int>(params.number_of_buckets_Z) + iy * static_cast<long long unsigned int>( params.number_of_buckets_Z) + iz;

			if (index_bucket >= 0 && index_bucket < params.number_of_buckets){
				if(occupied[index_bucket] == pSourceLocal.label){
					if(pSourceLocal.label == PointType::obstacle){
						sum_good_hits ++;
					}
				}
			}
		}
		device_particles[index_of_particle].overlap = sum_good_hits;
	}
}

cudaError_t cudaCountOverlaps (
		int threads,
		Point *device_points,
		size_t points_size,
		char* occupied,
		size_t occupied_size,
		Grid3DParams params,
		Particle *device_particles,
		size_t device_particles_size)
{
	int blocks = device_particles_size / threads + 1;

		kernel_cudaCountOverlaps<<< blocks, threads>>> (
				device_points,
				points_size,
				occupied,
				occupied_size,
				params,
				device_particles,
				device_particles_size);

	return cudaGetLastError();
}
