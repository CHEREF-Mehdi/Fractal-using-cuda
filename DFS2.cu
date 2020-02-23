
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono> 
#include <armadillo>

using namespace std; 
using namespace std::chrono; 
const short maxThreadPerblock=1024;

arma::Mat<float> Triangle = {{1.0, 0.0, 0.0},
                               {0.0, 1.0, 0.0},
                               {0.0, 0.0, 1.0}};
vector<arma::Mat<float>> transfMat{
                                    {{1.0, 0.5, 0.5},
                                    {0.0, 0.5, 0.0},
                                    {0.0, 0.0, 0.5}},
                                    {{0.5, 0.0, 0.0},
                                    {0.5, 1.0, 0.5},
                                    {0.0, 0.0, 0.5}},
                                    {{0.5, 0.0, 0.0},
                                    {0.0, 0.5, 0.0},
                                    {0.5, 0.5, 1.0}}
                                    };
const unsigned short dim=3; //nbr transformation
const short level=7; //nbr iteration

const unsigned short sizeV=9;
#define sizeTL (27)

const float h_v[sizeV]={1.0, 0.0, 0.0, 
						0.0, 1.0, 0.0, 
						0.0, 0.0, 1.0
						};

const short h_tlSize[dim]={0,9,18};
const float h_tl[sizeTL] = {1.0, 0.5, 0.5,
							0.0, 0.5, 0.0,
							0.0, 0.0, 0.5,//T0
    						0.5, 0.0, 0.0,
     						0.5, 1.0, 0.5,
     						0.0, 0.0, 0.5,//T1
   							0.5, 0.0, 0.0,
    						0.0, 0.5, 0.0,
     						0.5, 0.5, 1.0 //T2
							};

__constant__ float d_v[sizeV];//device verteses
__constant__ float d_tl[sizeTL];//device transformation list
__constant__ short d_offsetT[dim];
__constant__ short d_sizeV;

__global__ void DFSkernel(short level,unsigned short dim, unsigned int Bi,size_t offset){	
	size_t n=threadIdx.x + blockIdx.x * blockDim.x + offset;	
	unsigned short T;
	float *poly=new float[d_sizeV];
	float *p=new float[d_sizeV];
	memcpy(p, d_v, sizeof(float)*d_sizeV);
	
	while(level>=0){
		T=n/Bi;
		//if(threadIdx.x + blockIdx.x * blockDim.x + offset==11)printf("%d %d\n",T,d_offsetT[T]);

		for (short r = 0; r < 3 ; r++)
		{
			for (short c = 0; c < 3 ; c++)
			{			
				
				poly[r*3+c]=0;	
				for (short k = 0; k < 3 ; k++)
				{
					poly[r*3+c]+=d_tl[d_offsetT[T]+r*3+k] * p[k*3+c];
				}
			}			
		}

		n=n%Bi;
		Bi=Bi/dim;
		level--;
		memcpy(p, poly, sizeof(float)*d_sizeV);				
	}

	if(threadIdx.x + blockIdx.x * blockDim.x + offset==0){
		
		for (short i = 0; i < 3 ; i++)
		{
			for (short j = 0; j < 3 ; j++)
			{
				printf("%.4f, ",poly[i*3+j]);
			}
			printf("\n");
		}
	}
	
	//printf("thread %d\n",threadIdx.x + blockIdx.x * blockDim.x + offset);
	
}

cudaError_t DFS(unsigned int threadPerblock,unsigned int block,unsigned int Bi,size_t offset,unsigned short mode);

int main()
{
	size_t threads=pow(dim,level);
	unsigned int blocks= threads/maxThreadPerblock;
	unsigned short mode=threads%maxThreadPerblock;
	size_t offset=0;
	unsigned int threadPerblock;

	if(blocks==0){
		blocks=1;
		threadPerblock=threads;		
	}else{
		threadPerblock=maxThreadPerblock;
		if(mode!=0){
			offset=blocks*maxThreadPerblock;
		}
	}
	/*arma::Mat<float> res;
	res=transfMat[0]*Triangle;
	res=transfMat[0]*res;
	res=transfMat[0]*res;
	std::cout << res << std::endl;*/

	std::cout << "Iteration : " << level << std::endl;
	std::cout << "nbr Transformations : " << dim << std::endl;	
	std::cout << "Total Thread : " << threads << std::endl;
	std::cout << "nbr Block : " << blocks << std::endl;	
	std::cout << "Nbr Thread/Block : " << threadPerblock << std::endl;			
	std::cout << "mode : " << mode << std::endl;
	std::cout << "offset : " << offset << std::endl;	
	std::cout << std::endl;	
	
	auto start = high_resolution_clock::now(); 
	DFS(threadPerblock,blocks,threads/dim,offset,mode);
	auto stop = high_resolution_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
	cout <<"\n\nExecution time : " << duration.count() << "\n\n";	
		
}

cudaError_t DFS(unsigned int threadPerblock,unsigned int block,unsigned int Bi,size_t offset,unsigned short mode) {
	/*short* result = 0;
	size_t N=pow(I,B);*/
	cudaError_t cudaStatus;

	/*cudaStatus = cudaMalloc((void**)& result, combinations_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}*/
	
	cudaMemcpyToSymbol(d_v, h_v, sizeof(float)*sizeV);
	cudaMemcpyToSymbol(d_tl, h_tl, sizeof(float)*sizeTL);
	cudaMemcpyToSymbol(d_offsetT, h_tlSize, sizeof(short)*dim);
	cudaMemcpyToSymbol(d_sizeV, &sizeV, sizeof(short));

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpyToSymbol error: %s\n", cudaGetErrorString(cudaStatus));
	    //goto Error;
	}

	DFSkernel<<<block,threadPerblock >>> (level-1,dim,Bi,0);
	
	if(offset!=0){
		DFSkernel<<<1,mode >>> (level-1,dim,Bi,offset);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    //goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	    //goto Error;
	}
	

	/*cudaStatus = cudaMemcpy(combinations, result, combinations_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    //goto Error;
	}*/
	
	// for (int i = 0; i < itr_size; i++) {
	// 	if(i%size_Col==0) std::cout << std::endl;
	// 	std::cout << combinations[i] << " ";
	// }
	

	//std::cout << "\n\nEnd.\n";

 	//Error:
	//cudaFree(result);

	return cudaStatus;
}