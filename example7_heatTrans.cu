#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
//-I/media/hdd/lbl_trainData/softwarePackage/opencv2413/include  -I/media/hdd/lbl_trainData/softwarePackage/opencv2413/include/opencv  -L/media/hdd/lbl_trainData/softwarePackage/opencv2413/lib -lopencv_core  -lopencv_highgui  -lopencv_imgproc

#define DIM (1000)
#define HALFDIM (DIM/2)


__device__ void copySource(unsigned char*img_data,unsigned char *heat_source)
{
	int x=threadIdx.x + blockDim.x*blockIdx.x;
	int y=threadIdx.y + blockDim.y*blockIdx.y;
	int offset=y*DIM+x;
	if(heat_source[offset]!=0) img_data[offset]=heat_source[offset];
}

__device__ void trans(unsigned char*img_data,unsigned char *tmp)
{
	int x=threadIdx.x + blockDim.x*blockIdx.x;
	int y=threadIdx.y + blockDim.y*blockIdx.y;
	int offset=y*DIM+x;
	if(x==0&&y==0) 
	{
		int down=(y+1)*DIM+x;
		int right=y*DIM+x+1;
		tmp[offset]=img_data[down]+img_data[right]-img_data[offset];
	}
	else if(x==0&&y==DIM-1) 
	{
		int up=(y-1)*DIM+x;
		int right=y*DIM+x+1;
		tmp[offset]=img_data[up]+img_data[right]-img_data[offset];
	}
	else if(x==DIM-1&&y==DIM-1) 
	{
		int up=(y-1)*DIM+x;
		int left=y*DIM+x-1;
		tmp[offset]=img_data[up]+img_data[left]-img_data[offset];
	}
	else if(x==DIM-1&&y==0) 
	{
		int down=(y+1)*DIM+x;
		int left=y*DIM+x-1;
		tmp[offset]=img_data[down]+img_data[left]-img_data[offset];
	}//********************************
	else if(x==0&&y!=0&&y!=DIM-1) 
	{
		int up=(y-1)*DIM+x;
		int down=(y+1)*DIM+x;
		int right=y*DIM+x+1;
		tmp[offset]=img_data[down]+img_data[up]+img_data[right]-2*img_data[offset];
	}
	else if(x==DIM-1&&y!=0&&y!=DIM-1) 
	{
		int up=(y-1)*DIM+x;
		int down=(y+1)*DIM+x;
		int left=y*DIM+x-1;
		tmp[offset]=img_data[down]+img_data[up]+img_data[left]-2*img_data[offset];
	}
	else if(x!=0&&x!=DIM-1&&y==0) 
	{
		int down=(y+1)*DIM+x;
		int left=y*DIM+x-1;
		int right=y*DIM+x+1;
		tmp[offset]=img_data[down]+img_data[right]+img_data[left]-2*img_data[offset];
	}
	else if(x!=0&&x!=DIM-1&&y==DIM-1) 
	{
		int up=(y-1)*DIM+x;
		int left=y*DIM+x-1;
		int right=y*DIM+x+1;
		tmp[offset]=img_data[up]+img_data[right]+img_data[left]-2*img_data[offset];
	}
	else
	{
		int down=(y+1)*DIM+x;
		int up=(y-1)*DIM+x;
		int left=y*DIM+x-1;
		int right=y*DIM+x+1;
		tmp[offset]=img_data[up]+img_data[down]+img_data[right]+img_data[left]-3*img_data[offset];
	}
}

__global__ void heatTrans(void*img_data,void *heat_source,int time)
{
	void*tmp=NULL;
	cudaMalloc((void**)&tmp,DIM*DIM*sizeof(unsigned char));
	for(int i=0;i<time;i++)
	{
		copySource((unsigned char*)tmp,(unsigned char*)heat_source);
		trans((unsigned char*)tmp,(unsigned char*)img_data);
	}

	
}


int main()
{
	cv::Mat heatSource = cv::Mat::zeros(DIM,DIM,CV_8UC1);
	cv::rectangle(heatSource,cv::Rect(HALFDIM,HALFDIM,180,70),cv::Scalar(180),-1);
	cv::rectangle(heatSource,cv::Rect(70,80,80,70),cv::Scalar(250),-1);
	cv::Mat img = cv::Mat::zeros(DIM,DIM,CV_8UC1);
	unsigned char *dev_img=NULL,*dev_source=NULL;
	cudaMalloc((void**)&dev_img,DIM*DIM*sizeof(unsigned char));
	cudaMalloc((void**)&dev_source,DIM*DIM*sizeof(unsigned char));
	cudaMemcpy(dev_source,heatSource.data,DIM*DIM*sizeof(unsigned char),cudaMemcpyHostToDevice);
	dim3 gridShape((32+DIM)/32,(32+DIM)/32);
	dim3 blockShape(32,32);
	
	int time=10;
	
	heatTrans<<<gridShape,blockShape>>>((void*)dev_img,(void*)dev_source,time);
	
	cudaMemcpy(img.data,dev_img,DIM*DIM*sizeof(unsigned char),cudaMemcpyDeviceToHost);
	
	cudaFree(dev_img);
	cv::imwrite("1.jpg",img);
	return 0;
}