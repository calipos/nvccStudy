#include "cuda_fp16.h"
#include "device_functions.h"
#include <cublas_v2.h>
#include <cuda.h>
//#include <helper_cuda.h>
//#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;
//#include <cudaCode.h">

// /usr/local/cuda-8.0/bin/nvcc test_cublas.cu -I/usr/local/cuda-8.0/include  -gencode arch=compute_61,code=sm_61  -L/usr/local/cuda-8.0/lib64  -lcublas


//size为转换前float数据个数，转换后由size/2个half2存储所有数据
typedef unsigned uint;

union FP32
{
    uint u;
    float f;
    struct
    {
        uint Mantissa : 23;
        uint Exponent : 8;
        uint Sign : 1;
    };
};
union FP16
{
    unsigned short u;
    struct
    {
        uint Mantissa : 10;
        uint Exponent : 5;
        uint Sign : 1;
    };
};
float half_to_float(__half hf)
{
    FP16 h = *((FP16*)&hf);

    static const FP32 magic = { 113 << 23 };
    static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    uint exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o.f;
}
__half float_to_half(float fl)
{
    FP32 f32infty = { 255 << 23 };
    FP32 f16max = { (127 + 16) << 23 };
    FP32 magic = { 15 << 23 };
    FP32 expinf = { (255 ^ 31) << 23 };
    uint sign_mask = 0x80000000u;
    FP16 o = { 0 };

    FP32 f = *((FP32*)&fl);

    uint sign = f.u & sign_mask;
    f.u ^= sign;

    if (!(f.f < f32infty.u)) // Inf or NaN
        o.u = f.u ^ expinf.u;
    else
    {
        if (f.f > f16max.f) f.f = f16max.f;
        f.f *= magic.f;
    }

    o.u = f.u >> 13; // Take the mantissa bits
    o.u |= sign >> 16;
    return *((__half*)&o);
}

__global__ void float2HalfVec(float *src, __half *des, int size_)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size_ ; i += stride)
    {
        des[i] = __float2half(src[i]); 
    }
}
__global__ void half2FloatVec(__half *src, float *des, int size_)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size_ ; i += stride)
		des[i] = __half2float(src[i]);
}


__global__ void myHalf2Add(const __half *a,const  __half *b, __half *c, int size_)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size_ ; i += stride)
    {
        c[i] = __hadd(a[i],b[i]);
    }
}

__global__ void myHalf2Add_2float2half(const __half *a,const  __half *b, __half *c, int size_)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	for (int i = idx; i<size_ ; i += stride)
    {
        c[i] =__float2half(__half2float(a[i])+__half2float(b[i]));
    }
}


void showDeviceHalf(__half*data,int count)
{
    float *show_dev;
    cudaMalloc((void**)&show_dev,count*sizeof(float));
    float *show=(float*)malloc(count*sizeof(float));
    half2FloatVec<<<128,128>>>(data,show_dev,count);
    cudaMemcpy(show,show_dev,count*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<count;i++)
    {std::cout<<show[i]<<std::endl;}
    free(show);
    cudaFree(show_dev);
}
void showDevice(float*data,int count)
{
    float *show=(float*)malloc(count*sizeof(float));
    cudaMemcpy(show,data,count*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<count;i++)
    {std::cout<<show[i]<<std::endl;}
    free(show);
}


void float2HalfVec_c(float*vectDev1,__half*vectDev1_half,int count)
{
    float2HalfVec<<<2, 16 >>>(vectDev1,vectDev1_half,count);
}

void gemm32(float*in1,float*in2,float*out)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alf_float=1.0;
    const float bta_float=0.0;
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,2,4,3,
    &alf_float,
    in2,2,
    in1,3,
    &bta_float,
    out,2);
}
void gemm16(__half*in1,__half*in2,__half*out)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    const __half alf = float_to_half(1.0);
    const __half bet = float_to_half(0.0);
    cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,2,4,3,
    &alf,
    in2,2,
    in1,3,
    &bet,
    out,2);
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float vect1[12]={1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1};//4*3
    float vect2[6]={1.1,2.1,3.1,4.1,5.1,6.1};//3*2   out=4*2
    float *vectDev1, *vectDev2, *resDev;
    cudaMalloc((void**)&vectDev1,12*sizeof(float));
    cudaMalloc((void**)&vectDev2,6*sizeof(float));
    cudaMalloc((void**)&resDev,8*sizeof(float));
    cudaMemcpy(vectDev1,vect1,12*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vectDev2,vect2,6*sizeof(float),cudaMemcpyHostToDevice);
    
    showDevice(vectDev1,12);
    float alpha=2.0;
    cublasSscal(handle, 12, &alpha, vectDev1,  1);
    showDevice(vectDev1,12);
    
    return 0;
}

