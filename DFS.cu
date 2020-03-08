
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono> 

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

using namespace std; 
using namespace std::chrono; 

__global__ void fill_cases(short* result, size_t start, size_t value){
	size_t i= threadIdx.x+start;
	result[i] = value;
}

__global__ void fillByLevel(short* result, size_t dim, size_t level, size_t fill_size, size_t fill_offset, size_t global_offset) {
	if (level <= 0) return;

	short value = threadIdx.x;
	size_t start = fill_size * value + fill_offset + global_offset;
	size_t end = start + fill_size;
	for (size_t i = start; i < end; i++) result[i] = value;
	fillByLevel <<<1, dim >> > (result, dim, level - 1, fill_size / dim, start, global_offset);
	__syncthreads();
}

__global__ void DFSKernel(short *result, size_t dim, size_t level, size_t _itr_size) {
	short value = threadIdx.x;
	size_t fill_size = powf(dim, level - 1);
	size_t start = fill_size * value;
	size_t end = start + fill_size;
	for (size_t i = start; i < end; i++) result[i] = value;
	
    
	fillByLevel << <1, dim >> > (result, dim, level - 1, fill_size / dim, start, powf(dim, level));
	__syncthreads();
}

cudaError_t DFS(short*, size_t, size_t, size_t, size_t);

int main()
{
	const size_t dim = 3; //M
	const size_t level = 11; //N

	short* combinations;
	size_t itr_size = pow(dim, level) * level;
	size_t combinations_size = sizeof(short) * itr_size;

	std::cout << "Iteration : " << level << std::endl;
	std::cout << "nbr Transformations : " << dim << std::endl;
	std::cout << "Array length : " << itr_size << std::endl;
	std::cout << "memory size : " << combinations_size << std::endl;

	combinations = (short*)malloc(combinations_size);

	if(combinations == NULL) printf("\n\n ############ Memory allocation failed ############\n\n");		
	else
	{
		auto start = high_resolution_clock::now(); 
		DFS(combinations, dim, level, combinations_size, itr_size);
		auto stop = high_resolution_clock::now(); 
		auto duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
		cout <<"\n\nExecution time : " << duration.count() << "\n\n";
	}
	
	
}

cudaError_t DFS(short* combinations, size_t dim, size_t level, size_t combinations_size, size_t itr_size) {
	short* result = 0;
	size_t size_Col=pow(dim,level);
	cudaError_t cudaStatus;

	
	// auto start = std::chrono::high_resolution_clock::now(); 
	cudaStatus = cudaMalloc((void**)& result, combinations_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}
	

	DFSKernel<<<1, dim >>> (result, dim, level, itr_size);
	

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    //goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	    //goto Error;
	}

	// auto stop = high_resolution_clock::now(); 
	// auto duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
	// cout <<"\n\nExecution time : " << duration.count() << "\n\n";
	

	cudaStatus = cudaMemcpy(combinations, result, combinations_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    //goto Error;
	}
	
	/*for (int i = 0; i < itr_size; i++) {
		if(i%size_Col==0) std::cout << std::endl;
		std::cout << combinations[i] << " ";
	}*/
	

	//std::cout << "\n\nEnd.\n";

 	//Error:
	cudaFree(result);

	return cudaStatus;
}