#include <iostream>


__global__ void kernel(){}

int main()
{
    kernel<<<1,1>>>();
    std::cout<<123<<std::endl;
    return 0;
}