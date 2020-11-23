#include <iostream>
#define min(a,b) (a<b?a:b)
#define N (33*10)
const int threadPerBlock = 256;
const int blocksPerGrid = min(32, (N+threadPerBlock)/threadPerBlock);

__global__ void dot(float*a,float*b,float*c)
{
	__shared__ float cache[threadPerBlock];
	int tid=threadIdx.x+blockDim.x*blockIdx.x;
	int cacheIdx=threadIdx.x;
	float tmp=0;
	while(tid<N)
	{
		tmp+=a[tid]*b[tid];
		tid+=blockDim.x*gridDim.x;
	}
	cache[cacheIdx]=tmp;
	
	__syncthreads();
	
	int i=blockDim.x/2;
	while(i!=0)
	{
		if(threadIdx.x<i)
		{
			cache[cacheIdx]=cache[cacheIdx]+cache[cacheIdx+i];
			__syncthreads();//线程发散，使得syncthreads的运行变得糟糕
		}
		__syncthreads();
		i=i/2;
	}
	if(i==0 && threadIdx.x==0)
	{
		c[blockIdx.x]=cache[0];
	}
}



int main()
{
	float *a=NULL;
	float *b=NULL;
	int c=0;
	float *patital_c=NULL;
	float *dev_a=NULL;
	float *dev_b=NULL;
	float *dev_patital_c=NULL;
	a=(float*)malloc(N*sizeof(float));
	b=(float*)malloc(N*sizeof(float));
	patital_c=(float*)malloc(blocksPerGrid*sizeof(float));
	
	cudaMalloc((void**)&dev_a,N*sizeof(float));
	cudaMalloc((void**)&dev_b,N*sizeof(float));
	cudaMalloc((void**)&dev_patital_c,blocksPerGrid*sizeof(float));
	
	double result=0;
	for(int i=0;i<N;i++)
	{
		a[i]=i;
		b[i]=2*i;
		result+=2*i*i;
	}
	cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice);
	
	dot<<<blocksPerGrid,threadPerBlock>>>(dev_a,dev_b,dev_patital_c);
	cudaMemcpy(patital_c,dev_patital_c,blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost);
	
	for(int i=0;i<blocksPerGrid;i++)
	{
		c+=int(patital_c[i]);
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_patital_c);
	std::cout<<c<<std::endl;
	std::cout<<result<<std::endl;
	
	return 0;
}