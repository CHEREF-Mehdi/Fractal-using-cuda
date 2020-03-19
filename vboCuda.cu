
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cstdlib>
#include <chrono> 
#include <math.h>
#include <iostream>

using namespace std;
using namespace std::chrono; 

const short maxThreadPerblock=1024;
struct cudaGraphicsResource *cuda_vbo_resource;
GLuint points_vbo;
float* d_vbo_ptr = 0;

const unsigned short dim=3; //nbr transformation
const short level=5; //nbr iteration
const unsigned short sizeV=9; //size of polygone 
#define sizeTL (27) //size of transformation

const float h_v[sizeV]={1.0, 0.0, 0.0, 
						0.0, 1.0, 0.0, 
						0.0, 0.0, 1.0
						};
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
const short h_tlSize[dim]={0,9,18};

__constant__ float d_v[sizeV];//device verteses
__constant__ float d_tl[sizeTL];//device transformation list
__constant__ short d_offsetT[dim];
__constant__ short d_sizeV;


__global__ void IFSkernel(float *ver,short level,unsigned short dim, unsigned int Bi,size_t offset);

cudaError_t DFS(int thread, unsigned int threadPerblock,unsigned int block,unsigned int Bi,size_t offset,unsigned short mode);

void setUpCamera();

int main(void)
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

	// initialize a VBO
	points_vbo = 0;
	// generate 1 VBO buffer
	glGenBuffers(1, &points_vbo); 
	// bind points_vbo to GL_ARRAY_BUFFER. 
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	// locate the memory without initialize the values  
	glBufferData(GL_ARRAY_BUFFER, threads*9 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    
    

    std::cout << "Iteration : " << level << std::endl;
    std::cout << "nbr Transformations : " << dim << std::endl;	
    std::cout << "Total Thread : " << threads << std::endl;
    std::cout << "nbr Block : " << blocks << std::endl;	
    std::cout << "Nbr Thread/Block : " << threadPerblock << std::endl;			
    std::cout << "mode : " << mode << std::endl;
    std::cout << "offset : " << offset << std::endl;	    
    
    auto start = high_resolution_clock::now(); 
    DFS(threads,threadPerblock,blocks,threads/dim,offset,mode);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
    cout <<"\nExecution time : " << duration.count() << "\n";
    
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

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {        
        glClear(GL_COLOR_BUFFER_BIT);
                        
        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, threads*3); // Starting from vertex 0; 3 vertices for each triangle              

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

__global__ void IFSkernel(float *ver,short level,unsigned short dim, unsigned int Bi,size_t offset){	
	size_t N=threadIdx.x + blockIdx.x * blockDim.x + offset;
	size_t n=N;	
	unsigned short T;
	short nbrVertex=d_sizeV/3;
	float *poly=new float[d_sizeV];
	float *p=new float[d_sizeV];

	memcpy(p, d_v, sizeof(float)*d_sizeV);	
	
	while(level>=0){
		T=n/Bi;

		for (short r = 0; r < nbrVertex ; r++)
		{
			for (short c = 0; c < 3 ; c++)
			{							
				poly[r*3+c]=0;	
				for (short k = 0; k < 3 ; k++)
					poly[r*3+c]+=d_tl[d_offsetT[T]+r*3+k] * p[k*3+c];				
			}			
		}

		n=n%Bi;
		Bi=Bi/dim;
		level--;
		memcpy(p, poly, sizeof(float)*d_sizeV);				
	}
	//insert vertices in vbo
	for(short i=0;i<nbrVertex;i++)
	    for(short j=0;j<3;j++) ver[N*d_sizeV+i*3+j]=poly[i+j*3];	 	
}


cudaError_t DFS(int threads,unsigned int threadPerblock,unsigned int block,unsigned int Bi,size_t offset,unsigned short mode) {	

	cudaError_t cudaStatus;	
	
	cudaMemcpyToSymbol(d_v, h_v, sizeof(float)*sizeV);
	cudaMemcpyToSymbol(d_tl, h_tl, sizeof(float)*sizeTL);
	cudaMemcpyToSymbol(d_offsetT, h_tlSize, sizeof(short)*dim);
	cudaMemcpyToSymbol(d_sizeV, &sizeV, sizeof(short));

	//connet cuda_vbo_resource to points_vbo
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, points_vbo, cudaGraphicsMapFlagsNone);  
	//give access authority of points_vbo to cuda
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    
    size_t num_bytes;  
	//"verteses" points to the GPU memory data store of VBO (points_vbo) maped by cuda_vbo_resource 
    cudaGraphicsResourceGetMappedPointer((void **)&d_vbo_ptr, &num_bytes, cuda_vbo_resource); 

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpyToSymbol error: %s\n", cudaGetErrorString(cudaStatus));
	    goto Error;
	}

	IFSkernel<<<block,threadPerblock >>> (d_vbo_ptr,level-1,dim,Bi,0);
	
	if(offset!=0){
		IFSkernel<<<1,mode >>> (d_vbo_ptr,level-1,dim,Bi,offset);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    goto Error;
	}	

 	Error:
	cudaFree(d_v);
	cudaFree(d_tl);
	cudaFree(d_offsetT);

	return cudaStatus;
}

void setUpCamera(){
    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    gluPerspective(30, 1, 0.1, 500);
    gluLookAt(2, 2, 2, 0, 0.2, 0, 0, 1, 0);
    glPopMatrix();
}