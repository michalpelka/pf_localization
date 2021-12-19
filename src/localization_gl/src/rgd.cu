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
