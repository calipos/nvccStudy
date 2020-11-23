#include "cuda_fp16.h"
#include "device_functions.h"
#include <cublas_v2.h>
#include <cuda.h>
//#include <helper_cuda.h>
//#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;
//#include <cudaCode.h">

// /usr/local/cuda-8.0/bin/nvcc fp16.cu -I/usr/local/cuda-8.0/include  -gencode arch=compute_61,code=sm_61  -L/usr/local/cuda-8.0/lib64  -lcublas


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

int main_test3216()
{
    float vect1[12]={1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1};//4*3
    float vect2[6]={1.1,2.1,3.1,4.1,5.1,6.1};//3*2   out=4*2
    float *vectDev1, *vectDev2, *resDev;
    cudaMalloc((void**)&vectDev1,12*sizeof(float));
    cudaMalloc((void**)&vectDev2,6*sizeof(float));
    cudaMalloc((void**)&resDev,8*sizeof(float));
    cudaMemcpy(vectDev1,vect1,12*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vectDev2,vect2,6*sizeof(float),cudaMemcpyHostToDevice);
    gemm32(vectDev1,vectDev2,resDev);
    showDevice(resDev,8);
    
    __half *vectDev1_half, *vectDev2_half, *resDev_half;
    cudaMalloc((void**)&vectDev1_half,12*sizeof(__half));
    cudaMalloc((void**)&vectDev2_half,6*sizeof(__half));
    cudaMalloc((void**)&resDev_half,8*sizeof(__half));
    
    //float2HalfVec_c(vectDev1,vectDev1_half,12);
    //float2HalfVec_c(vectDev2,vectDev2_half,6);
    float2HalfVec<<<2,128>>>(vectDev1,vectDev1_half,12);
    float2HalfVec<<<2,128>>>(vectDev2,vectDev2_half,6);
    gemm16(vectDev1_half,vectDev2_half,resDev_half);
    showDeviceHalf(resDev_half,8);
    //showDeviceHalf(vectDev1_half,12);
    //showDeviceHalf(vectDev2_half,6);
    
    return 0;
}
int main32()
{
    float vect1[12]={1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1};//4*3
    float vect2[6]={1.1,2.1,3.1,4.1,5.1,6.1};//3*2   out=4*2
    float *vectDev1, *vectDev2, *resDev;
    cudaMalloc((void**)&vectDev1,12*sizeof(float));
    cudaMalloc((void**)&vectDev2,6*sizeof(float));
    cudaMalloc((void**)&resDev,8*sizeof(float));
    cudaMemcpy(vectDev1,vect1,12*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vectDev2,vect2,6*sizeof(float),cudaMemcpyHostToDevice);
    gemm32(vectDev1,vectDev2,resDev);
    showDevice(resDev,8);
    return 0;
}


int main16()
{

    float vect1[12]={1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1};//4*3
    float vect2[6]={1.1,2.1,3.1,4.1,5.1,6.1};//3*2   out=4*2
    float *vectDev1, *vectDev2, *resDev;
    cudaMalloc((void**)&vectDev1,12*sizeof(float));
    cudaMalloc((void**)&vectDev2,6*sizeof(float));
    cudaMalloc((void**)&resDev,8*sizeof(float));
    cudaMemcpy(vectDev1,vect1,12*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vectDev2,vect2,6*sizeof(float),cudaMemcpyHostToDevice);
    
    __half *vectDev1_half, *vectDev2_half, *resDev_half;
    cudaMalloc((void**)&vectDev1_half,12*sizeof(__half));
    cudaMalloc((void**)&vectDev2_half,6*sizeof(__half));
    cudaMalloc((void**)&resDev_half,8*sizeof(__half));
    
    //float2HalfVec_c(vectDev1,vectDev1_half,12);
    //float2HalfVec_c(vectDev2,vectDev2_half,6);
    float2HalfVec<<<2,128>>>(vectDev1,vectDev1_half,12);
    float2HalfVec<<<2,128>>>(vectDev2,vectDev2_half,6);
    
    //gemm16(vectDev1_half,vectDev2_half,resDev_half);
    //showDeviceHalf(resDev_half,8);
    showDeviceHalf(vectDev1_half,12);
    showDeviceHalf(vectDev2_half,6);
    return 0;
}

int main3()
{
    cudaDeviceProp prop;
    int whichDev;
    cudaGetDevice(&whichDev);
    cudaGetDeviceProperties(&prop,whichDev);
    std::cout<<"# "<<prop.deviceOverlap<<std::endl;
    
    const int blocks=128;
    const int threads=128;
    size_t size_=blocks*threads*16;
    size_=16;
    float*vect1=new float[size_];
    float*vect2=new float[size_];
    float*res=new float[size_];
    float*res_half2float2half=new float[size_];
    for(int i=0;i<size_;i++)
    {
        vect2[i]=vect1[i]=0.90/(i+1.1);
    }
    float*vectDev1,*vectDev2,*resDev;
    cudaMalloc((void**)&vectDev1,size_*sizeof(float));
    cudaMalloc((void**)&vectDev2,size_*sizeof(float));
    cudaMalloc((void**)&resDev,size_*sizeof(float));
    cudaMemcpy(vectDev1,vect1,size_*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vectDev2,vect2,size_*sizeof(float),cudaMemcpyHostToDevice);
    
    __half *vectHalfDev1,*vectHalfDev2,*resHalfDev1;
    cudaMalloc((void**)&vectHalfDev1,size_*sizeof(float)/2);
    cudaMalloc((void**)&vectHalfDev2,size_*sizeof(float)/2);
    cudaMalloc((void**)&resHalfDev1,size_*sizeof(float)/2);

float2HalfVec <<<128, 128 >>>(vectDev1, vectHalfDev1, size_);
float2HalfVec <<<128, 128 >>>(vectDev2, vectHalfDev2, size_);

    float *show_dev;
    cudaMalloc((void**)&show_dev,16*sizeof(float));
    float *show=(float*)malloc(16*sizeof(float));
    half2FloatVec<<<128,128>>>(vectHalfDev1,show_dev,16);
    
    cudaMemcpy(show,show_dev,16*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<16;i++)
    {std::cout<<"*"<<show[i]<<std::endl;}

    cudaFree(show_dev);
    cudaMemcpy(show,vectDev1,16*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<16;i++)
    {std::cout<<"-"<<show[i]<<std::endl;}
    free(show);
myHalf2Add << <128, 128 >> > (vectHalfDev1, vectHalfDev2, resHalfDev1, size_);
half2FloatVec << <128, 128 >> >(resHalfDev1, resDev, size_);

cudaMemcpy(res, resDev, size_ * sizeof(float), cudaMemcpyDeviceToHost);


	for (int i = 0; i < 16; i++)//打印出前64个结果，并与CPU结果对比
	{
		cout << vect1[i] << " + " << vect2[i] << " = " << vect1[i] + vect2[i] << "  ?  " << res[i]<< " - "<<std::endl;;
	}
    
	delete[] vect1;
	delete[] vect2;
	delete[] res;
    delete[] res_half2float2half;
	(cudaFree(vectDev1));
	(cudaFree(vectDev2));
	(cudaFree(resDev));
	(cudaFree(vectHalfDev1));
	(cudaFree(vectHalfDev2));
	(cudaFree(resHalfDev1));
    return  0;    
}


int main2()
{
     cublasHandle_t handle;
     cublasCreate(&handle);
     
    float a_host[60]={1.,1.,2.,2.,3.,3.};
    float b_host[20]={0.1,0.1};
    float c_host[3];
    __half c_host_half[3];
    float *a_dev,*b_dev,*c_dev;
    
    
    cudaMalloc((void**)&a_dev,6*sizeof(float));
    cudaMalloc((void**)&b_dev,2*sizeof(float));
    cudaMalloc((void**)&c_dev,3*sizeof(float));
    cudaMemcpy(a_dev,a_host,6*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev,b_host,2*sizeof(float),cudaMemcpyHostToDevice);
    //showDevice(a_dev,60);
    //showDevice(b_dev,20);

     __half *a_dev_half,*b_dev_half,*c_dev_half;
     cudaMalloc((void**)&a_dev_half,6*sizeof(__half));
     cudaMalloc((void**)&b_dev_half,2*sizeof(__half));
     cudaMalloc((void**)&c_dev_half,3*sizeof(__half));

     float2HalfVec<<<2, 16 >>>(a_dev,a_dev_half,60);
     float2HalfVec<<<2, 16 >>>(b_dev,b_dev_half,20);
    //showDeviceHalf(a_dev_half,60);
    //showDeviceHalf(b_dev_half,20);
    //return 0;
     const __half alf = float_to_half(1.0);
     const __half bet = float_to_half(0.0);
     const __half *alpha = &alf;
     const __half *beta = &bet;

     cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,3,2,
     alpha,
     b_dev_half,1,
     a_dev_half,2,
     beta,
     c_dev_half,1);
     

     
     cudaMemcpy(c_host_half,c_dev_half,3*sizeof(__half),cudaMemcpyDeviceToHost);
    std::cout<< half_to_float(c_host_half[0])<<std::endl;
    std::cout<< half_to_float(c_host_half[1])<<std::endl;
    std::cout<< half_to_float(c_host_half[2])<<std::endl;
    
    
     const float alf_float=1.0;
     const float bta_float=0.0;
     cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,3,2,
     &alf_float,
     b_dev,1,
     a_dev,2,
     &bta_float,
     c_dev,1);
    cudaMemcpy(c_host,c_dev,3*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<< (c_host[0])<<std::endl;
    std::cout<< (c_host[1])<<std::endl;
    std::cout<< (c_host[2])<<std::endl;
    
    return  0;    
}


int main()
{
     cublasHandle_t handle;
     cublasCreate(&handle);
     
    float a_host[60]={1.2345678,2.3456789,3.45678901,4.56789012,5.67890123,6.78901234};
    float b_host[20]={0.123,0.234};
    float c_host[3];
    //__half c_host_half[3];
    float *a_dev,*b_dev,*c_dev;
    
    
    cudaMalloc((void**)&a_dev,6*sizeof(float));
    cudaMalloc((void**)&b_dev,2*sizeof(float));
    cudaMalloc((void**)&c_dev,3*sizeof(float));
    cudaMemcpy(a_dev,a_host,6*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev,b_host,2*sizeof(float),cudaMemcpyHostToDevice);

     const float alf_float=1.0;
     const float bta_float=0.0;
     cublasSgemv(handle,CUBLAS_OP_T,2,3,
     &alf_float,
     a_dev,2,
     b_dev,1,
     &bta_float,
     c_dev,1);
     std::cout<<"------------cublasSgemv-----------"<<std::endl;
    showDevice(c_dev,3);

    

     __half *a_dev_half,*b_dev_half,*c_dev_half;
     cudaMalloc((void**)&a_dev_half,6*sizeof(__half));
     cudaMalloc((void**)&b_dev_half,2*sizeof(__half));
     cudaMalloc((void**)&c_dev_half,3*sizeof(__half));

     float2HalfVec<<<2, 16 >>>(a_dev,a_dev_half,60);
     float2HalfVec<<<2, 16 >>>(b_dev,b_dev_half,20);
    //showDeviceHalf(a_dev_half,60);
    //showDeviceHalf(b_dev_half,20);
    //return 0;
     const __half alf = float_to_half(1.0);
     const __half bet = float_to_half(0.0);
     const __half *alpha = &alf;
     const __half *beta = &bet;

     cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,3,2,
     alpha,
     b_dev_half,1,
     a_dev_half,2,
     beta,
     c_dev_half,1);
     std::cout<<"------------cublasHgemm-----------"<<std::endl;
    showDeviceHalf(c_dev_half,3);
    std::cout<<"------------half_to_float-----------"<<std::endl;
    __half *c_host_half=(__half*)malloc(3*sizeof(__half));
    cudaMemcpy(c_host_half,c_dev_half,3*sizeof(__half),cudaMemcpyDeviceToHost);
    std::cout<<half_to_float(c_host_half[0])<<std::endl;
    std::cout<<half_to_float(c_host_half[1])<<std::endl;
    std::cout<<half_to_float(c_host_half[2])<<std::endl;

    std::cout<<half_to_float(float_to_half(-1.2e-8))<<std::endl;
    return  0;    
}