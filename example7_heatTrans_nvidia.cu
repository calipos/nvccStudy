#include <iostream>
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#define DIM (1024)

typedef unsigned char uchar;



__global__ void copySource(uchar*source,uchar*dest)
{
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;
	int offset=x+y*blockDim.x*gridDim.x;
	if(source[offset]!=0) dest[offset]=source[offset];
}

__global__ void trans(uchar*in,uchar*out,float speed=0.2)
{
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;
	int offset=x+y*blockDim.x*gridDim.x;
	int up=y==0?offset:offset-blockDim.x*gridDim.x;
	int down=y==DIM-1?offset:offset+blockDim.x*gridDim.x;
	int left=x==0?offset:offset-1;
	int right=x==DIM-1?offset:offset+1;
	
	float tmp_inc=speed*(1.0*in[up]+in[down]+in[left]+in[right]-4.0*in[offset]);
	int tmp=0;
	if (tmp_inc>=speed && tmp_inc<=1)
		tmp_inc=1.0;
	tmp=tmp_inc+in[offset];
	
	if (tmp<1)out[offset]=0;
	else if (tmp>255)out[offset]=255;
	else out[offset]=tmp;

	
}

typedef struct dataBlock
{
	uchar* dev_img;
	uchar* dev_out;
	uchar* dev_source;
	cudaEvent_t start,end;
	float totaltime;
	float frames;

}dataBlock;

int main()
{
	dataBlock db;
	uchar* dev_img;
	uchar* dev_out;
	uchar* dev_source;
	cudaMalloc((void**)&dev_img,DIM*DIM*sizeof(uchar));
	cudaMalloc((void**)&dev_out,DIM*DIM*sizeof(uchar));
	cudaMalloc((void**)&dev_source,DIM*DIM*sizeof(uchar));
	cv::Mat img = cv::Mat::zeros(DIM,DIM,CV_8UC1);
	cv::rectangle(img,cv::Rect(DIM/2,DIM/2,380,270),cv::Scalar(250),-1);
	cv::rectangle(img,cv::Rect(100,200,170,380),cv::Scalar(100),-1);

	cudaMemcpy(dev_source,img.data,DIM*DIM*sizeof(uchar),cudaMemcpyHostToDevice);
	db.dev_img=dev_img;
	db.dev_out=dev_out;
	db.dev_source=dev_source;
	
	cudaEventCreate(&db.start);
	cudaEventCreate(&db.end);
	cudaEventRecord(db.start,0);
	dim3 blocks(DIM/16,DIM/16);
	dim3 threads(16,16);
	for(int i=0;i<1000;i++)
	{
		copySource<<<blocks,threads>>>(db.dev_source,db.dev_img);

		trans<<<blocks,threads>>>(db.dev_img,db.dev_out,0.26);
		cudaMemcpy(db.dev_img,db.dev_out,DIM*DIM*sizeof(uchar),cudaMemcpyDeviceToDevice);
		cudaMemcpy(img.data,db.dev_img,DIM*DIM*sizeof(uchar),cudaMemcpyDeviceToHost);
		cv::imshow("123",img);
		cv::waitKey(15);
		std::cout<<i<<std::endl;
		
	}
	cv::imwrite("1.jpg",img);
	cudaEventRecord(db.end,0);
	cudaEventSynchronize(db.end);
	
	return 0;
}






