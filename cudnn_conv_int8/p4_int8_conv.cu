
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <ctime>
//#include <char>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


///usr/local/cuda-8.0/bin/nvcc p4_int8_conv.cu -std=c++11    -I/usr/local/cuda-8.0/include    -I/media/hdd/lbl_trainData/softwarePackage/cudnn-8.0-linux-x64-v7/include  -L/usr/local/cuda-8.0/lib64  -lcublas -lcudart -L/media/hdd/lbl_trainData/softwarePackage/cudnn-8.0-linux-x64-v7/lib64  -lcudnn  -gencode arch=compute_61,code=sm_61  


/** Error handling from https://developer.nvidia.com/cuDNN */
#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    cudaDeviceReset();                                                         \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);              \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCudaErrors(status)                                                \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)


int main() {
    cudnnHandle_t cudnnHandle;
    checkCudaErrors(cudaSetDevice(0));
    checkCUDNN(cudnnCreate(&cudnnHandle));
  cudnnTensorDescriptor_t dataTensor;
  checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 2, 5, 3,3));

  return 0;
}
