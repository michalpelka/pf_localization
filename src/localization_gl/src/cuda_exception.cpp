#include "cuda_exception.h"
#include "helper_cuda.h"
#include <sstream>

void throw_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess){
	std::cout << "CUDA error: " << code << " " << _cudaGetErrorEnum(code) << " " << file << " " << line << std::endl;


	std::stringstream ss;
    ss << "cuda: " << file << "(" << line << ")";
    std::string file_and_line = ss.str();
    throw MyCudaError(code, file_and_line, _cudaGetErrorEnum(code));
  }
}

