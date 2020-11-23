#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include "fp16_conversion.h"
#include "util.h"

using namespace std;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
///usr/local/cuda-8.0/bin/nvcc p4_igemm.cu -std=c++11    -I/usr/local/cuda-8.0/include    -I/media/hdd/lbl_trainData/softwarePackage/cudnn-8.0-linux-x64-v7/include  -L/usr/local/cuda-8.0/lib64  -lcublas -lcudart -L/media/hdd/lbl_trainData/softwarePackage/cudnn-8.0-linux-x64-v7/lib64  -lcudnn  -gencode arch=compute_61,code=sm_61  

int main_test(int argc, char ** argv){


  int min_m_k_n = 16;
  int max_m_k_n = 4096*8;
  int repeats = 10;
  int verbose = 1;

  cout << "\nrunning cublasSgemm test\n" << endl;
  
  if(verbose) 
    cout << "running with" 
	 << " min_m_k_n: " << min_m_k_n
	 << " max_m_k_n: " << max_m_k_n
	 << " repeats: " << repeats
	 << endl;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  if(verbose) cout << "allocating device variables" << endl;
  
  // Allocate 3 arrays on CPU
  
  int8_t *h_A = (int8_t *)malloc(max_m_k_n * max_m_k_n * sizeof(int8_t));
  int8_t *h_B = (int8_t *)malloc(max_m_k_n * max_m_k_n * sizeof(int8_t));
  int8_t *h_C = (int8_t *)malloc(max_m_k_n * max_m_k_n * sizeof(int8_t));
  
  //CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
  //CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
  //CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);

    // Allocate 3 arrays on GPU
    int8_t *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(int8_t)));
    checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(int8_t)));
    checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(int8_t)));
    
    checkCuda(cudaMemcpy(d_A,h_A,max_m_k_n * max_m_k_n * sizeof(int8_t),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B,h_B,max_m_k_n * max_m_k_n * sizeof(int8_t),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C,h_C,max_m_k_n * max_m_k_n * sizeof(int8_t),cudaMemcpyHostToDevice));
    
    int lda, ldb, ldc, m, n, k;
    int alf = 1;
    int bet = 0;
    int *alpha = &alf;
    int *beta = &bet;
  

  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int size = min_m_k_n; size <= max_m_k_n; size=size*2){

    m=n=k=size;
    lda = m;
    ldb = k;
    ldc = m;

    cudaEventRecord(start, 0);


    for(int rep = 0; rep < repeats; rep++){

           stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                            m, n, k, 
                            alpha, 
                            d_A, CUDA_R_8I, lda, 
                            d_B, CUDA_R_8I, ldb, 
                            beta, 
                            d_C, CUDA_R_32I, ldc,
                            CUDA_R_32I, CUBLAS_GEMM_DFALT);
          
          checkCublas(stat);
          assert(!cudaGetLastError());
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 1000.0f;

    if(stat != CUBLAS_STATUS_SUCCESS){
      cerr << "cublasSgemm failed" << endl;
      exit(1);
    }

    //assert(!cudaGetLastError());

    cout << "int8_t; size "

         << size << " average: " << elapsed/repeats << " s "<< endl;

  }

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C);
      
  return 0;
}

template<typename Dtype>
void cpu_filler(Dtype*container,int height, int width)
{
    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++)
            container[width*i+j]=Dtype((i-1)*width+j+1);
}
template<typename Dtype>
void show_cpu_filler(Dtype*container,int height, int width)
{
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            std::cout<<(int)(container[width*i+j])<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"-----------"<<std::endl;
}

int igemm()
{

    int m=3;//这个好像不必要是4*n
    int n=12;
    int k=8;

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));
  //4*3  3*5
  int8_t *h_A = (int8_t *)malloc(m * k * sizeof(int8_t));
  int8_t *h_B = (int8_t *)malloc(k * n * sizeof(int8_t));
  int *h_C_32 = (int *)malloc(m * n * sizeof(int));
  
  cpu_filler(h_A, m, k);
  cpu_filler(h_B, k, n);

  show_cpu_filler(h_A, m, k);
  show_cpu_filler(h_B, k, n);



    int8_t *d_A, *d_B;
    int * d_C_32;
    checkCuda(cudaMallocManaged(&d_A, m * k * sizeof(int8_t)));
    checkCuda(cudaMallocManaged(&d_B, k * n * sizeof(int8_t)));
    checkCuda(cudaMallocManaged(&d_C_32, m * n * sizeof(int)));
    
    checkCuda(cudaMemcpy(d_A,h_A,m * k * sizeof(int8_t),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B,h_B,k * n * sizeof(int8_t),cudaMemcpyHostToDevice));
    

    int alf=1;
    int bet=0;




    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    n, m, k, 
                    &alf, 
                    d_B, CUDA_R_8I, n, 
                    d_A, CUDA_R_8I, k, 
                    &bet, 
                    d_C_32, CUDA_R_32I, n,
                    CUDA_R_32I, CUBLAS_GEMM_DFALT);
          

    checkCuda(cudaMemcpy(h_C_32,d_C_32,m * n * sizeof(int),cudaMemcpyDeviceToHost));
    std::cout<<h_C_32[0]<<" "<<h_C_32[1]<<std::endl;
    show_cpu_filler(h_C_32, m, n);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C_32);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C_32);
      
  return 0;
}

void gemm32()
{
  
  
    int m=3;//这个好像不必要是4*n
    int n=12;
    int k=8;

  //4*3  3*5
  float *h_A = (float *)malloc(m * k * sizeof(float));
  float *h_B = (float *)malloc(k * n * sizeof(float));
  float *h_C_32 = (float *)malloc(m * n * sizeof(float));
  
  cpu_filler(h_A, m, k);
  cpu_filler(h_B, k, n);

  show_cpu_filler(h_A, m, k);
  show_cpu_filler(h_B, k, n);



    float *d_A, *d_B;
    float * d_C_32;
    checkCuda(cudaMallocManaged(&d_A, m * k * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_B, k * n * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_C_32, m * n * sizeof(float)));
    
    checkCuda(cudaMemcpy(d_A,h_A,m * k * sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B,h_B,k * n * sizeof(float),cudaMemcpyHostToDevice));
    

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alf_float=1.0;
    const float bta_float=0.0;
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,
    &alf_float,
    d_B,n,
    d_A,k,
    &bta_float,
    d_C_32,n);
    
    
    checkCuda(cudaMemcpy(h_C_32,d_C_32,m * n * sizeof(int),cudaMemcpyDeviceToHost));
    std::cout<<h_C_32[0]<<" "<<h_C_32[1]<<std::endl;
    show_cpu_filler(h_C_32, m, n);
}


int getNewDim(int n, int k,int*newN,int*newK)
{
	if(n%4==0 && k%4==0)
	{
		*newN = n;
		*newK = k;
		return 0;
	}	
	*newN = n%4==0? n : (n/4+1)*4;
	*newK = k%4==0? k : (k/4+1)*4;
	return 1;
}

//N是大矩阵的数量
template<typename Dtype>
__global__ void copyData(int N, Dtype*dataIn, Dtype*dataOut, int row, int col, int newRow, int newCol)
{
	//dataIn1  : m*k
	//dataIn2  : n*k
	//dataOut1 : m*newK
	//dataOut2 : newN*newK
	CUDA_KERNEL_LOOP(idx,N){
		int i = idx / newCol;
		int j = idx % newCol;
		if(i>=row || j>=col)  dataOut[idx]=0;
		else 
		{
			int pos = j+i*col;
			dataOut[idx] = dataIn[pos];
			
		}
	}
}

//N是大矩阵的数量
template<typename Dtype>
__global__ void copyData_back(int N, Dtype*dataIn, Dtype*dataOut, int row, int col, int newRow, int newCol)
{
	//dataIn1  : m*k
	//dataIn2  : n*k
	//dataOut1 : m*newK
	//dataOut2 : newN*newK
	CUDA_KERNEL_LOOP(idx,N){
		int i = idx / newCol;
		int j = idx % newCol;
		if(i>=row || j>=col)  continue;
		else 
		{
			int pos = j+i*col;
			dataIn[idx] = dataOut[pos];
			
		}
	}
}


int igemm(int m, int n, int k)
{
    // int m=3;//这个好像不必要是4*n
    // int n=12;
    // int k=8;
  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));
  //4*3  3*5
  int8_t *h_A = (int8_t *)malloc(m * k * sizeof(int8_t));
  int8_t *h_B = (int8_t *)malloc(k * n * sizeof(int8_t));
  int *h_C_32 = (int *)malloc(m * n * sizeof(int));  
  
  cpu_filler(h_A, m, k);
  cpu_filler(h_B, k, n);
  //show_cpu_filler(h_A, m, k);
  //show_cpu_filler(h_B, k, n);

	int8_t *d_A, *d_B;
	int * d_C_32;
	checkCuda(cudaMallocManaged(&d_A, m * k * sizeof(int8_t)));
	checkCuda(cudaMallocManaged(&d_B, k * n * sizeof(int8_t)));
	checkCuda(cudaMallocManaged(&d_C_32, m * n * sizeof(int)));
	checkCuda(cudaMemcpy(d_A,h_A,m * k * sizeof(int8_t),cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_B,h_B,k * n * sizeof(int8_t),cudaMemcpyHostToDevice));
	
	int newK=0;
	int newN=0;
	int8_t *d_A_new, *d_B_new;
	int * d_C_32_new;
	int needReshape = getNewDim(n,k,&newN,&newK);

	if(needReshape>0) 
	{
		checkCuda(cudaMallocManaged(&d_A_new, m * newK * sizeof(int8_t)));
		checkCuda(cudaMallocManaged(&d_B_new, newK * newN * sizeof(int8_t)));
		checkCuda(cudaMallocManaged(&d_C_32_new, m * newN * sizeof(int)));
		int bigger_count1=m*newK;
		int bigger_count2=newK*newN;
		copyData<int8_t><<<2,256>>>(bigger_count1,d_A,d_A_new,m,k,m,newK);
		copyData<int8_t><<<2,256>>>(bigger_count2,d_B,d_B_new,k,n,newK,newN);
	}
	else
	{
		d_A_new = d_A;
		d_B_new = d_B;
		d_C_32_new=d_C_32;
	}

	int alf=1;
	int bet=0;

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
					newN, m, newK, 
					&alf, 
					d_B_new, CUDA_R_8I, newN, 
					d_A_new, CUDA_R_8I, newK, 
					&bet, 
					d_C_32_new, CUDA_R_32I, newN,
					CUDA_R_32I, CUBLAS_GEMM_DFALT);		  
	if(needReshape>0)
	{
		int biggerCount = m * newN;
		copyData_back<int><<<2,256>>>(biggerCount,d_C_32,d_C_32_new,m,n,m,newN);
		cudaFree(d_A_new);
		cudaFree(d_B_new);
		cudaFree(d_C_32_new);
	}
	checkCuda(cudaMemcpy(h_C_32,d_C_32,m * n * sizeof(int),cudaMemcpyDeviceToHost));
	std::cout<<h_C_32[0]<<" "<<h_C_32[1]<<std::endl;
	show_cpu_filler(h_C_32, m, n);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C_32);

  // Free CPU memory
  free(h_A);
  free(h_B);
  free(h_C_32);
      
  return 0;
}

void gemm32(int m, int n, int k)
{
  
  
    // int m=3;//这个好像不必要是4*n
    // int n=12;
    // int k=8;

  //4*3  3*5
  float *h_A = (float *)malloc(m * k * sizeof(float));
  float *h_B = (float *)malloc(k * n * sizeof(float));
  float *h_C_32 = (float *)malloc(m * n * sizeof(float));
  
  cpu_filler(h_A, m, k);
  cpu_filler(h_B, k, n);

  //show_cpu_filler(h_A, m, k);
  //show_cpu_filler(h_B, k, n);



    float *d_A, *d_B;
    float * d_C_32;
    checkCuda(cudaMallocManaged(&d_A, m * k * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_B, k * n * sizeof(float)));
    checkCuda(cudaMallocManaged(&d_C_32, m * n * sizeof(float)));
    
    checkCuda(cudaMemcpy(d_A,h_A,m * k * sizeof(float),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B,h_B,k * n * sizeof(float),cudaMemcpyHostToDevice));
    

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alf_float=1.0;
    const float bta_float=0.0;
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,
    &alf_float,
    d_B,n,
    d_A,k,
    &bta_float,
    d_C_32,n);
    
    
    checkCuda(cudaMemcpy(h_C_32,d_C_32,m * n * sizeof(int),cudaMemcpyDeviceToHost));
    std::cout<<h_C_32[0]<<" "<<h_C_32[1]<<std::endl;
    show_cpu_filler(h_C_32, m, n);
}


int main()
{
  igemm(3,4,5);
  gemm32(3,4,5);
  igemm(3,4,12);
  gemm32(3,4,12);
  igemm(4,4,3);
  gemm32(4,4,3);
  return 0;
}