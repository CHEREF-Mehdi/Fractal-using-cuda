
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cstdlib>
#include <chrono> 
#include <math.h>
#include <iostream>

using namespace std;
using namespace std::chrono; 

const short maxThreadPerblock=1024;

const unsigned short dim=3; //nbr transformation
const short level=9; //nbr iteration

const unsigned short sizeV=9;
float *ver;
#define sizeTL (27)

const float h_v[sizeV]={1.0, 0.0, 0.0, 
						0.0, 1.0, 0.0, 
						0.0, 0.0, 1.0
						};

float trgl[sizeV]= {1.0, 0.0, 0.0, 
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


__global__ void DFSkernel(float *ver,short level,unsigned short dim, unsigned int Bi,size_t offset){	
	size_t N=threadIdx.x + blockIdx.x * blockDim.x + offset;
	size_t n=N;	
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

	for(short i=0;i<3;i++)
	    for(short j=0;j<3;j++) ver[N*9+i*3+j]=poly[i+j*3];	 
	
}

cudaError_t DFS(float *ver,int thread, unsigned int threadPerblock,unsigned int block,unsigned int Bi,size_t offset,unsigned short mode);


void setUpCamera();

int main(void)
{
    size_t threads=pow(dim,level);
	unsigned int blocks= threads/maxThreadPerblock;
	unsigned short mode=threads%maxThreadPerblock;
	size_t offset=0;
	unsigned int threadPerblock;
	ver=(float*)malloc((threads*9) * sizeof(float));
    if (ver!=NULL){
        if(blocks==0){
            blocks=1;
            threadPerblock=threads;		
        }else{
            threadPerblock=maxThreadPerblock;
            if(mode!=0){
                offset=blocks*maxThreadPerblock;
            }
        }

        std::cout << "Iteration : " << level << std::endl;
        std::cout << "nbr Transformations : " << dim << std::endl;	
        std::cout << "Total Thread : " << threads << std::endl;
        std::cout << "nbr Block : " << blocks << std::endl;	
        std::cout << "Nbr Thread/Block : " << threadPerblock << std::endl;			
        std::cout << "mode : " << mode << std::endl;
        std::cout << "offset : " << offset << std::endl;	
        std::cout << std::endl;	
        
        auto start = high_resolution_clock::now(); 
        DFS(ver,threads,threadPerblock,blocks,threads/dim,offset,mode);
        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
        cout <<"\n\nExecution time : " << duration.count() << "\n\n";
    }else
    {
        cout <<"\n\n Malloc error \n\n";
    }
	

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    /* Init glew */
    if(glewInit() != GLEW_OK)
        std::cout << "glewInit error" << std::endl;
    
    setUpCamera();

    //std::cout << "GL_VERSION : " << glGetString(GL_VERSION) << std::endl;

	GLuint points_vbo = 0;
	glGenBuffers(1, &points_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glBufferData(GL_ARRAY_BUFFER, threads*9 * sizeof(float), ver, GL_DYNAMIC_DRAW);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
        glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
        );
        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, threads*3); // Starting from vertex 0; 3 vertices total -> 1 triangle
        glDisableVertexAttribArray(0);        

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

cudaError_t DFS(float *ver,int threads,unsigned int threadPerblock,unsigned int block,unsigned int Bi,size_t offset,unsigned short mode) {
	float* result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)& result, threads*9* sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}
	
	cudaMemcpyToSymbol(d_v, h_v, sizeof(float)*sizeV);
	cudaMemcpyToSymbol(d_tl, h_tl, sizeof(float)*sizeTL);
	cudaMemcpyToSymbol(d_offsetT, h_tlSize, sizeof(short)*dim);
	cudaMemcpyToSymbol(d_sizeV, &sizeV, sizeof(short));

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpyToSymbol error: %s\n", cudaGetErrorString(cudaStatus));
	    //goto Error;
	}

	DFSkernel<<<block,threadPerblock >>> (result,level-1,dim,Bi,0);
	
	if(offset!=0){
		DFSkernel<<<1,mode >>> (result,level-1,dim,Bi,offset);
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
	

	cudaStatus = cudaMemcpy(ver, result, threads*9*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    //goto Error;
	}
	
	/*for (int i = 0; i < threads*9; i+=9) {
		for(int j=0;j<9;j++) std::cout << ver[i+j] << " | ";
		std::cout  << std::endl;
	}*/
	
	//std::cout << "\n\nEnd.\n";

 	//Error:
	//cudaFree(result);

	return cudaStatus;
}

void setUpCamera(){
    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    gluPerspective(30, 1, 0.1, 500);
    gluLookAt(2, 2, 2, 0, 0.2, 0, 0, 1, 0);
    glPopMatrix();
}