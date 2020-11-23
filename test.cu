#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <sstream>
#include <random>


cudaDeviceProp getCudaDeviceProperties(int deviceIdx = 0) {
	cudaSetDevice(deviceIdx);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceIdx);
	return deviceProp;
}




///在GPU上跑的函数，被称为kernel function
__global__ void productArray_bt_kernel(float *pa, float *pb, float *pResult, int N) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N)
		pResult[idx] = pa[idx] * pb[idx];
}

///kernel function 的辅助函数，用于自动分配内存、验证结果等;此处将内存的分配、拷贝放在了外部！因为同一参数要执行好多次
void productArray_bt(float *pa, float *pb, float *pResult, int N, int threadNum = 32) throw(std::string) {
	cudaError_t cudaStatus;
	int blockNum = (N - 1) / threadNum + 1;
	dim3 bd(blockNum, 1, 1);
	productArray_bt_kernel << <blockNum, threadNum >> > (pa, pb, pResult, N);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		std::stringstream ss;
		ss << "productArray_bt_kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\n\tblockNum=" << blockNum << ";\tthreadNum=" << threadNum << ";\t";
		std::string errStr = ss.str();
		//std::cerr << errStr << std::endl;
		throw errStr;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::stringstream ss;
		ss << "cudaDeviceSynchronize returned error code" << cudaStatus << "after launching addKernel!" << "\n\tblockNum=" << blockNum << ";\tthreadNum=" << threadNum << ";\t";
		std::string errStr = ss.str();
		//std::cerr <<errStr<< std::endl;
		throw errStr;
	}
}


void TestProductSpeed() {
	float *pa, *pb, *pResult;			// host copies
	float *d_a, *d_b, *d_c;	// device copies

	int N = 5;



	for (N = 1; N <= (1 << 20); N *= 2) {
		std::cout << "\n\n**************数组长度为  " << N << "  的测试\n";
		int size = sizeof (float)* N;


		// Allocate space for device copies of pa, pb, pResult
		cudaMalloc((void **)&d_a, size);
		cudaMalloc((void **)&d_b, size);
		cudaMalloc((void **)&d_c, size);


		pa = new float[N];
		pb = new float[N];
		pResult = new float[N];
		for (int i = 0; i < N; ++i) {
			pa[i] = i;
			pb[i] = i * 10;
		}



		// Copy inputs to device
		cudaMemcpy(d_a, pa, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, pb, size, cudaMemcpyHostToDevice);

		//std::cout << "blockDim.x=" << blockDim.x<<std::endl;
		// Launch add() kernel on GPU

		for (int threadNum = 1; threadNum <= 4096; threadNum *= 4) {
			try{
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);
				{
					//统计GPU耗时的代码段
					productArray_bt(d_a, d_b, d_c, N, threadNum);
				}
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				float costtime;
				cudaEventElapsedTime(&costtime, start, stop);

				std::cout << "数组长度=" << N << ";\t" << "treadNum=" << threadNum << ";\t" << "点积用时：" << costtime / 1000 << "s" << std::endl;



				// Copy result back to host
				cudaMemcpy(pResult, d_c, size, cudaMemcpyDeviceToHost);


				//验证结果的正确性
				for (int i = 1; i < N; i *= 2) {
					//std::cout << "pa[i]=" << pa[i] << std::endl;
					if (pResult[i] != pa[i] * pb[i])
						std::cout << "错误: " << "i=" << i << ";\ti*10i=" << pResult[i] << std::endl;

				}


			}
			catch (std::string s){
				std::cout << "异常：" << s << std::endl;
				std::cout << "\t" << "数组长度=" << N << ";\t" << "treadNum=" << threadNum << ";\t" << std::endl;

			}
			catch (...){
				std::cout << "未知的异常类型" << std::endl;
				std::cout << "\t\t" << "数组长度=" << N << ";\t" << "treadNum=" << threadNum << ";\t" << std::endl;
			}


			std::cout << std::endl;

		}

		// Cleanup
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);

		delete[]pa;
		delete[]pb;
		delete[]pResult;
	}

}





#define printExp(x) std::cout<< #x <<" = "<< x <<std::endl;

int main(void) {
	std::cout << __FILE__ << std::endl;

	TestProductSpeed();
	return 0;
}
