
#include <GL/glew.h>
#include <GL/glut.h>
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
const short level=4; //nbr iteration

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

	/*for(short i=0;i<9;i++){
		ver[N*9+i]=poly[i];
	}*/

	memcpy(ver+(N*9),p,sizeof(float)*d_sizeV);  
	/*if(N==26){
		
		for (short i = 0; i < 3 ; i++)
		{
			for (short j = 0; j < 3 ; j++)
			{
				printf("%.4f, ",ver[N*9+i*3+j]);
			}
			printf("\n");
		}
	}*/
	
	//printf("thread %d\n",threadIdx.x + blockIdx.x * blockDim.x + offset);
	
}

cudaError_t DFS(float *ver,int thread, unsigned int threadPerblock,unsigned int block,unsigned int Bi,size_t offset,unsigned short mode);

// menu item
#define MENU_SMOOTH 1
#define MENU_FLAT 0

double colors[1000] = {0};

int iterations = level, maxIteration = 15;
double zoom = 1;
int shading = GL_SMOOTH;

// Function prototypes
void generateColors();
double random(bool reset);
void keyboard(unsigned char key, int x, int y);
//void special(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void menu(int item);
void display();
void init();

void drawPolygone(GLfloat ** poly)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(shading);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 3; i++)
    {
        glColor3f(random(false), random(false), random(false));
        glVertex3fv(poly[i]);
    }
    glEnd();
}

void toGLfloatPoints(float *armapoly, int size)
{
	GLfloat **poly = (GLfloat **)malloc(3 * sizeof(GLfloat *));
	for (int i = 0; i < 3; i++) poly[i] = (GLfloat *)malloc(3 * sizeof(GLfloat));			
	
	for(int k=0;k<size;k+=9){
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++) poly[i][j] = armapoly[k+j*3+i];	
		}
		drawPolygone(poly);
	}
    
}



void display()
{
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(zoom, zoom, zoom);
    random(true); 
	//std::cout << (sizeof(ver)/sizeof(*ver))/9 << std::endl;
	//std::cout << ver[27*9] << std::endl;
	//toGLfloatPoints(ver,pow(dim,level)*9);
	
	
    glFlush();
}

int main(int argc, char **argv)
{

    size_t threads=pow(dim,level);
	unsigned int blocks= threads/maxThreadPerblock;
	unsigned short mode=threads%maxThreadPerblock;
	size_t offset=0;
	unsigned int threadPerblock;
	ver=(float*)malloc((threads*9) * sizeof(float));

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
	
    generateColors();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(600, 600);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Sierpinski Triangle");
    glutPositionWindow(100, 100);

    glutKeyboardFunc(keyboard);
    //glutSpecialFunc(special);
    glutMouseFunc(mouse);

    glutCreateMenu(menu);
    glutAddMenuEntry("Smooth shading", MENU_SMOOTH);
    glutAddMenuEntry("Flat shading", MENU_FLAT);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
	glewInit();
    init();
    glutDisplayFunc(display);
    glutMainLoop();

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


void generateColors()
{
    for (int i = 0; i < 1000; i++)
    {
        colors[i] = rand() / (double)RAND_MAX;
    }
}

double random(bool reset)
{
    static int curr = 0;
    if (reset)
    {
        curr = 0;
        return 0.0;
    }
    else
    {
        if (curr >= 1000)
            curr = 0;
        return colors[curr++];
    }
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case '+':
        if (iterations < maxIteration)
            iterations += 1;
        display();
        break;
    case '-':
        if (iterations > 0)
            iterations -= 1;
        display();
        break;
    case 'q':
        exit(0);
        break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if ((button == 3) || (button == 4)) // It's a wheel event
    {
        if (button == 3)
        {
            zoom += 0.5;
        }
        else if (button == 4)
        {
            if (zoom >= 1.5)
                zoom -= 0.5;
            else
                zoom = 1;
        }
        display();
    }
    else
    { // normal button event
        //if (button == GLUT_LEFT_BUTTON){

        if (state == GLUT_UP)
        {
            generateColors();
            display();
        }
    }
}

void menu(int item)
{
    switch (item)
    {
    case MENU_FLAT:
        shading = GL_FLAT;
        display();
        break;
    case MENU_SMOOTH:
        shading = GL_SMOOTH;
        display();
        break;
    }
}

void init()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glColor3f(0.0, 0.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluPerspective(30, 1, 0.1, 500);
    gluLookAt(2, 2, 2, 0, 0.2, 0, 0, 1, 0);
}
