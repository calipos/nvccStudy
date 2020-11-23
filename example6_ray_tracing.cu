
#include "opencv/cv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
//-I/media/hdd/lbl_trainData/softwarePackage/opencv2413/include  -I/media/hdd/lbl_trainData/softwarePackage/opencv2413/include/opencv  -L/media/hdd/lbl_trainData/softwarePackage/opencv2413/lib -lopencv_core  -lopencv_highgui  -lopencv_imgproc

#define DIM (500)
#define HALFDIM (DIM/2)
#define SOPHERE_NUM (200)
#define rnd(x) (rand()%100*0.01*x)

typedef struct sphere
{
	float r,g,b;
	float x,y,z;
	float rad,rSqr;
	__device__ float hitNearest(int ox,int oy,float*scale)
	{
		float distanceFromCenterSqr = ((ox-x)*(ox-x)+(oy-y)*(oy-y));
		if (distanceFromCenterSqr<=rSqr)
		{
			float dz=sqrt(rSqr-distanceFromCenterSqr);
			*scale=dz/rad;
			return z+dz;
		}
		else
		{
			scale=0;
			return -1;
		}
	}
	void show(){std::cout<<r<<", "<<g<<", "<<b<<", "<<x<<", "<<y<<", "<<z<<", "<<rad<<", "<<rSqr<<" "<<std::endl;}
}sphere;


__global__  void rayTracing(void*img_data,void*sphere_data)
{
	unsigned char*imgData1=(unsigned char*)img_data;
	unsigned char*imgData2=(unsigned char*)img_data+DIM*DIM;
	unsigned char*imgData3=(unsigned char*)img_data+DIM*DIM*2;

	int x =threadIdx.x+blockIdx.x*blockDim.x;
	int y =threadIdx.y+blockIdx.y*blockDim.y;
	int offset=x+DIM*y;
	float nearest=-1;
	for(int i=0;i<SOPHERE_NUM;i++)
	{
		float scale_=0;
		float distance = ((sphere*)sphere_data)[i].hitNearest(x-HALFDIM,y-HALFDIM,&scale_);
		if(distance>nearest)
		{
			nearest=distance;
			imgData1[offset]=scale_*((sphere*)sphere_data)[i].b;
			imgData2[offset]=scale_*((sphere*)sphere_data)[i].g;
			imgData3[offset]=scale_*((sphere*)sphere_data)[i].r;
		}
	}
}

int main()
{
	sphere*spheres=(sphere*)malloc(SOPHERE_NUM*sizeof(sphere));	
	for(int i=0;i<SOPHERE_NUM;i++)
	{
		spheres[i].x=rnd(500)-250;
		spheres[i].y=rnd(500)-250;
		spheres[i].z=rnd(500)-250;
		spheres[i].r=rnd(255);
		spheres[i].g=rnd(255);
		spheres[i].b=rnd(255);
		spheres[i].rad=rnd(100)+20;
		spheres[i].rSqr=spheres[i].rad*spheres[i].rad;
		spheres[i].show();
	}
	unsigned char *dev_sphere=NULL;
	cudaMalloc((void**)&dev_sphere,SOPHERE_NUM*sizeof(sphere));
	cudaMemcpy(dev_sphere,spheres,SOPHERE_NUM*sizeof(sphere),cudaMemcpyHostToDevice);
	free(spheres);
	
	
	cv::Mat img = cv::Mat::zeros(DIM,DIM,CV_8UC3);
	unsigned char *dev_img=NULL;
	cudaMalloc((void**)&dev_img,3*DIM*DIM*sizeof(unsigned char));
	dim3 gridShape(DIM/25,DIM/25);
	dim3 blockShape(25,25);
	rayTracing<<<gridShape,blockShape>>>((void*)dev_img,(void*)dev_sphere);
	
	
	//cudaMemcpy(img.data,dev_img,3*DIM*DIM*sizeof(unsigned char),cudaMemcpyDeviceToHost);

	unsigned char* tmp=(unsigned char*)malloc(3*DIM*DIM*sizeof(unsigned char));
	cudaMemcpy(tmp,dev_img,3*DIM*DIM*sizeof(unsigned char),cudaMemcpyDeviceToHost);
	int dim1_=DIM*DIM;
	int dim2_=2*dim1_;
	int pixel=0;
	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			img.at<cv::Vec3b>(i,j)[0]=tmp[pixel];
			img.at<cv::Vec3b>(i,j)[1]=tmp[pixel+dim1_];
			img.at<cv::Vec3b>(i,j)[2]=tmp[pixel+dim2_];
			pixel++;
		}
	}

	cudaFree(dev_sphere);
	cudaFree(dev_img);
	cv::imwrite("1.jpg",img);
		cv::imshow("123",img);
	cv::waitKey();
	return 0;
}