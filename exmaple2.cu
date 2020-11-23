#include <iostream>


__global__ void add(int a,int b, int*c)
{
    *c=a+b;
}

int main()
{
    int c=0;
    int *dev_c=NULL;
    std::cout<<"cudaMalloc ret = "<<cudaMalloc((void**)&dev_c,sizeof(int))<<std::endl;
	
	add<<<1,1>>>(2,7,dev_c);
	
    std::cout<<"cudaMemcpy ret = "<<cudaMemcpy(&c,dev_c,sizeof(int),cudaMemcpyDeviceToHost)<<std::endl;
	std::cout<<c<<std::endl;
	std::cout<<"cudaFree ret = "<<cudaFree(dev_c)<<std::endl;
    std::cout<<123<<std::endl;
    return 0;
}