#include <iostream>



int main()
{
	int count=0;
	std::cout<<"cudaGetDeviceCount ret = "<<cudaGetDeviceCount(&count)<<std::endl;
	std::cout<<"device count = "<<count<<std::endl;
	cudaDeviceProp prop;
	for(int i=0;i<count;i++)
	{
		
		std::cout<<"cudaGetDeviceProperties ret = "<<cudaGetDeviceProperties(&prop,i)<<std::endl;
		
		std::cout<<"---general device info for device "<<i<<"---"<<std::endl;
		std::cout<<"Name : "<<prop.name<<std::endl;
		std::cout<<"compute capability : "<<prop.major<<" "<<prop.minor<<std::endl;
		
		
	}
	
    std::cout<<123<<std::endl;
    return 0;
}