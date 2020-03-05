#include <GL/glut.h>
#include <armadillo>
#include <cuda.h>
#include <cstdlib>
#include <stack>
#include <chrono> 
#include <math.h>
#include <iostream>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

//files.associations

using namespace std;
using namespace std::chrono; 

// menu item
#define MENU_SMOOTH 1
#define MENU_FLAT 0

// N => iterations M => nbr of transformations
#define N 20
#define M 3

struct fractalLevel
{
    arma::Mat<GLfloat> poly;
    int iteration;
};

struct transformLevel
{
    vector<int> trans;
    int iteration;
};

arma::Mat<GLfloat> Triangle = {{0.0, 0.0, -1.0},
                               {1.0, 0.0, 1.0},
                               {0.0, 1.0, 1.0}};

arma::Mat<GLfloat> Triangle2 = {{1.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0}};

vector<arma::Mat<GLfloat>> transfMat{
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

double colors[1000] = {0};

int iterations = 9, maxIteration = 200;
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
GLfloat **toGLfloatPoints(arma::Mat<GLfloat> armapoly, int n_row);
void drawPolygone(GLfloat **poly, int n_row);
void displayPolygone(arma::Mat<GLfloat> trgl);

void dividePolygone(arma::Mat<GLfloat> poly, vector<arma::Mat<GLfloat>> TransfList, int iteration)
{
    if (iteration == 0)
    {
        //displayPolygone(poly);
    }
    else
    {
        for (int i = 0; i < TransfList.size(); i++)
            dividePolygone(TransfList[i] * poly, TransfList, iteration - 1);
    }
}

void dividePolygoneIterative(arma::Mat<GLfloat> poly, vector<arma::Mat<GLfloat>> TransfList, int iter)
{
    stack<fractalLevel> stk;
    fractalLevel level;

    while (true)
    {
        while (iter > 0)
        {
            iter--;
            level.iteration = iter;
            for (int i = TransfList.size() - 1; i > 0; i--)
            {
                level.poly = TransfList[i] * poly;
                stk.push(level);
            }

            poly = TransfList[0] * poly;
        }

        displayPolygone(poly);

        if (stk.empty())
            break;
        else
        {
            level = stk.top();
            stk.pop();
            poly = level.poly;
            iter = level.iteration;
        }
    }
}

void getListTransform(vector<vector<int>> &Tlist, int TlistSize, int iter)
{
    stack<transformLevel> stk;
    transformLevel level, l;
    arma::Mat<GLfloat> t;

    while (true)
    {
        while (iter > 0)
        {
            iter--;
            level.iteration = iter;
            for (int i = TlistSize - 1; i > 0; i--)
            {
                l = level;
                l.trans.push_back(i);
                stk.push(l);
            }

            level.trans.push_back(0);
        }

        Tlist.push_back(level.trans);

        if (stk.empty())
            break;
        else
        {
            level = stk.top();
            stk.pop();
            iter = level.iteration;
        }
    }
}

void dividePolygoneIterative2(arma::Mat<GLfloat> trgl, vector<arma::Mat<GLfloat>> TransfList, int iter)
{
    vector<vector<int>> Tlist;
    arma::Mat<GLfloat> t;

    getListTransform(Tlist, TransfList.size(), iter);

    /*for (int j = 0; j < Tlist.size(); j++)
    {
        t = trgl;
        for (auto i = Tlist[j].cbegin(); i != Tlist[j].cend(); ++i)
        {
            t = TransfList[*i] * t;
        }

        displayPolygone(t);
    }*/
}

__global__ void fillByLevel(int* result, size_t dim, size_t level, size_t fill_size, size_t fill_offset, size_t global_offset) {
	if (level <= 0) return;
	//fill buffer result by level
	int value = threadIdx.x;
	size_t start = fill_size * value + fill_offset + global_offset;
	size_t end = start + fill_size;
	//printf("start: %d == end: %d\n", start, end);
	for (size_t i = start; i < end; i++) {
		result[i] = value;
		//printf("%d\n", start);
	}
	
	fillByLevel <<<1, dim >> > (result, dim, level - 1, fill_size / dim, start, global_offset);
	
}

__global__ void DFSKernel(int *result, size_t dim, size_t level, size_t _itr_size) {
	int value = threadIdx.x;
	size_t fill_size = powf(dim, level - 1);
	size_t start = fill_size * value;
	size_t end = start + fill_size;
	//printf("start: %d == end: %d\n", start, end);
	for (size_t i = start; i < end; i++) {
		result[i] = value;
		//printf("%d\n", start);
	}
    
	fillByLevel << <1, dim >> > (result, dim, level - 1, fill_size / dim, start, powf(dim, level));
	__syncthreads();
}

cudaError_t DFS(int*, size_t, size_t, size_t, size_t);

void display()
{
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(zoom, zoom, zoom);
    random(true);

    cout <<"\n N = " << iterations <<"\n" << endl;

    auto start = high_resolution_clock::now(); 
    dividePolygone(Triangle, transfMat, iterations);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
    cout <<"Recursive : " << duration.count() << endl;

    start = high_resolution_clock::now(); 
    dividePolygoneIterative(Triangle2, transfMat, iterations);
    stop = high_resolution_clock::now(); 
    duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
    cout <<"Iterative 1 : " << duration.count() << endl;

    start = high_resolution_clock::now(); 
    dividePolygoneIterative2(Triangle2, transfMat, iterations);
    stop = high_resolution_clock::now(); 
    duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
    cout <<"Iterative 2 : " << duration.count() << endl;    

    glFlush();
}

int main(int argc, char **argv)
{
    const size_t dim = M; //M
	const size_t level = iterations; //N

	int* combinations;
	size_t itr_size = pow(dim, level) * level;    
	size_t combinations_size = sizeof(int) * itr_size;
    combinations = (int*)malloc(combinations_size);

    if(combinations == NULL) printf("\n\n ############ Memory allocation failed ############\n\n");		
	else
	{
		
        auto start = high_resolution_clock::now(); 
        DFS(combinations, dim, level, combinations_size, itr_size);
        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start)*pow(10,-6);
        cout <<"cuda : " << duration.count() << endl;
	}

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

    init();
    glutDisplayFunc(display);
    glutMainLoop();

    return 0;
}

cudaError_t DFS(int* combinations, size_t dim, size_t level, size_t combinations_size, size_t itr_size) {
	int* result = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		//print error message
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& result, combinations_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	DFSKernel<<<1, dim >>> (result, dim, level, itr_size);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	    goto Error;
	}

	cudaStatus = cudaMemcpy(combinations, result, combinations_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaMemcpy failed!");
	    goto Error;
	}

	// for (int i = 0; i < itr_size; i++) {
	// 	std::cout << combinations[i] << "\n";
	// }

	// std::cout << "End.";

    Error:
	    cudaFree(result);

	return cudaStatus;
}

void displayPolygone(arma::Mat<GLfloat> trgl)
{
    trgl = trgl.t();
    GLfloat **poly = toGLfloatPoints(trgl, trgl.n_rows);
    drawPolygone(poly, trgl.n_rows);
}

GLfloat **toGLfloatPoints(arma::Mat<GLfloat> armapoly, int n_row)
{
    GLfloat **poly = (GLfloat **)malloc(n_row * sizeof(GLfloat *));
    for (int i = 0; i < n_row; i++)
    {
        poly[i] = (GLfloat *)malloc(3 * sizeof(GLfloat));
        for (int j = 0; j < 3; j++)
        {
            poly[i][j] = armapoly(i, j);
        }
    }
    return poly;
}

void drawPolygone(GLfloat **poly, int n_row)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(shading);
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < n_row; i++)
    {
        glColor3f(random(false), random(false), random(false));
        glVertex3fv(poly[i]);
    }
    glEnd();
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
