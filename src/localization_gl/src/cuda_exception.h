#ifndef __CUDA_EXCEPTION_H__
#define __CUDA_EXCEPTION_H__



#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>


void throw_cuda_error(cudaError_t code, const char *file, int line);


class MyCudaError:public std::runtime_error
{
public:
	MyCudaError(int errCode, const std::string & errSrc, const std::string & errMsg)
		: std::runtime_error(errMsg), err(errCode),  source(errSrc) {

	}

	~MyCudaError(){};

	int err;
	std::string source;
};

#endif
