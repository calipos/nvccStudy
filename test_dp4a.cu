#include "cuda_fp16.h"
#include "device_functions.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>
#include <sm_61_intrinsics.h>
#include <vector_types.h>
using namespace std;


// /usr/local/cuda-8.0/bin/nvcc test_dp4a.cu -I/usr/local/cuda-8.0/include  -gencode arch=compute_61,code=sm_61  -L/usr/local/cuda-8.0/lib64  -lcublas


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

typedef union 
{
    int int32_data;
    struct 
    {
        char char1;
        char char2;
        char char3;
        char char4;
    }int8s_data;
}int32_8;


__global__ void  compute_int8_4(int count, int*in, int*weight, int*out)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x*blockDim.x;
    for (int i = idx; i<count ; i += stride)
    {
        out[i] = __dp4a(in[i],weight[i],0); 
    }
}


int main()
{
    

    
    // std::cout<<"sizeof(int32_8) : "<<sizeof(int32_8)<<std::endl;
    // int32_8 x;
    // x.int32_data=1;
    // std::cout<<"----------------"<<std::endl;
    // std::cout<<"x.int32_data=1"<<std::endl;
    // std::cout<<"x.int8s_data.char1 = "<<(int)x.int8s_data.char1<<std::endl;
    // std::cout<<"x.int8s_data.char2 = "<<(int)x.int8s_data.char2<<std::endl;
    // std::cout<<"x.int8s_data.char3 = "<<(int)x.int8s_data.char3<<std::endl;
    // std::cout<<"x.int8s_data.char4 = "<<(int)x.int8s_data.char4<<std::endl;
    // std::cout<<"----------------"<<std::endl;
    // std::cout<<"----------------"<<std::endl;
    // int32_8 y;
    // y.int8s_data.char1=1;
    // y.int8s_data.char2=1;
    // y.int8s_data.char3=0;
    // y.int8s_data.char4=0;
    // std::cout<<"x.int8s_data.char1 = 1"<<std::endl;
    // std::cout<<"x.int8s_data.char2 = 1"<<std::endl;
    // std::cout<<"x.int8s_data.char3 = 0"<<std::endl;
    // std::cout<<"x.int8s_data.char4 = 0"<<std::endl;
    // std::cout<<"y.int32_data = "<<y.int32_data<<std::endl;
    // std::cout<<"----------------"<<std::endl;
    
    
    // x.int8s_data.char1=10;
    // x.int8s_data.char2=20;
    // x.int8s_data.char3=30;
    // x.int8s_data.char4=40;
    // y.int8s_data.char1=1;
    // y.int8s_data.char2=2;
    // y.int8s_data.char3=3;
    // y.int8s_data.char4=4;
    
    char4 x[2],y[2];
    x[0].x=10;
    x[0].y=20;
    x[0].z=30;
    x[0].w=40;
    y[0].x=1;
    y[0].y=2;
    y[0].z=3;
    y[0].w=4;
        x[1].x=1;
    x[1].y=2;
    x[1].z=3;
    x[1].w=4;
    y[1].x=5;
    y[1].y=6;
    y[1].z=7;
    y[1].w=8;
    int *in, *weight, *out;
    cudaMalloc((void**)&in,2*sizeof(int));
    cudaMalloc((void**)&weight,2*sizeof(int));
    cudaMalloc((void**)&out,2*sizeof(int));
    cudaMemcpy(in,x,2*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(weight,y,2*sizeof(int),cudaMemcpyHostToDevice);
    compute_int8_4<<<1,2>>>(2,in,weight,out);
    int out_host[2];
    cudaMemcpy(&out_host,out,2*sizeof(int),cudaMemcpyDeviceToHost);
std::cout<<out_host[0]<<std::endl;
std::cout<<out_host[1]<<std::endl;
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // float vect1[12]={1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1};//4*3
    // float vect2[6]={1.1,2.1,3.1,4.1,5.1,6.1};//3*2   out=4*2
    // float *vectDev1, *vectDev2, *resDev;
    // cudaMalloc((void**)&vectDev1,12*sizeof(float));
    // cudaMalloc((void**)&vectDev2,6*sizeof(float));
    // cudaMalloc((void**)&resDev,8*sizeof(float));
    // cudaMemcpy(vectDev1,vect1,12*sizeof(float),cudaMemcpyHostToDevice);
    // cudaMemcpy(vectDev2,vect2,6*sizeof(float),cudaMemcpyHostToDevice);
    // showDevice(vectDev1,12);
    // float alpha=2.0;
    // cublasSscal(handle, 12, &alpha, vectDev1,  1);
    // showDevice(vectDev1,12);
    
    return 0;
}

