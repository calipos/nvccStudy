#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
//-I/media/hdd/lbl_trainData/softwarePackage/opencv2413/include  -I/media/hdd/lbl_trainData/softwarePackage/opencv2413/include/opencv  -L/media/hdd/lbl_trainData/softwarePackage/opencv2413/lib -lopencv_core  -lopencv_highgui  -lopencv_imgproc

#define DIM (100)
#define HALFDIM (50)

__device__ int julia(int x, int y)
{
	if (x+y>DIM ) return 1;
	else return -1;
	const float scale=1.5;
	float xj=scale*(1.0*x-HALFDIM)/HALFDIM;
	float yj=scale*(1.0*y-HALFDIM)/HALFDIM;
	for (int i=0;i<200;i++)
	{
		xj=xj*xj-yj*yj-0.8;
		yj=2*xj*yj+0.156;
		if(xj*xj+yj*yj>1000) return -1;
		else return 1;
	}
	return 1;
}

__global__  void drawMat(void*img_data)
{
	
	unsigned char*imgData=(unsigned char*)img_data;
	int x =blockIdx.x;
	int y =blockIdx.y;
	
	int isSatisfy = julia(x,y);
	if(isSatisfy>=0)
		imgData[y*gridDim.x+x]=255;
	else
		imgData[y*gridDim.x+x]=0;
}

int main()
{
	cv::Mat img = cv::Mat::zeros(DIM,DIM,CV_8UC1);
	unsigned char *dev_data=NULL;
	cudaMalloc((void**)&dev_data,DIM*DIM*sizeof(unsigned char));
	dim3 gridShape(DIM,DIM);
	drawMat<<<gridShape,1>>>((void*)dev_data);
	
	cudaMemcpy(img.data,dev_data,DIM*DIM*sizeof(unsigned char),cudaMemcpyDeviceToHost);
	
	cudaFree(dev_data);
	cv::imwrite("1.jpg",img);
	return 0;
}