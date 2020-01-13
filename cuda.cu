#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <cuda.h>

#include <armadillo>

#include <GL/glut.h>

using namespace arma;
using namespace std;

#define N (2042 * 2042)
#define THREADS_PER_BLOCK 512

GLfloat xRotated, yRotated, zRotated;

void init(void);
void DrawCube(void);
void animation(void);
void reshape(int x, int y);

void randomInts(int *a, int n)
{
  int i;
  for (i = 0; i < n; i++)
  {
    a[i] = rand()%(10000-100 + 1) + 100;
  }
}

void saveToFile(FILE *fp, int *a, int *b, int *c)
{
  for (int i = 0; i < 10; i++)
  {
    fprintf(fp, "%d + %d = %d\n", a[i], b[i],c[i]);
  }
}

__global__ 
void add(int *a, int *b, int *c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

int main(int argc, char **argv)
{
  
  cout << "Armadillo version: " << arma_version::as_string() << endl;
  
  
  int *a, *b, *c;
  int *d_a, *d_b, *d_c; 
  int size = N * sizeof(int);
  
  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Setup input values
  a = (int *)malloc(size);
  randomInts(a, N);
  b = (int *)malloc(size);
  randomInts(b, N);
  c = (int *)malloc(size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU
  add<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  FILE *fp;

  fp = fopen("result.txt","w");
  saveToFile(fp,a,b,c);
  fclose(fp);

  // Cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  glutInit(&argc, argv);
  //we initizlilze the glut. functions
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("glut openGl and cuda");

  //info version GLSL
  cout << "***** Info GPU *****" << std::endl;
  cout << "Fabricant : " << glGetString(GL_VENDOR) << std::endl;
  cout << "Carte graphique: " << glGetString(GL_RENDERER) << std::endl;
  cout << "Version : " << glGetString(GL_VERSION) << std::endl;
  cout << "Version GLSL : " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

  init();
  glutDisplayFunc(DrawCube);
  glutReshapeFunc(reshape);
  //Set the function for the animation.
  glutIdleFunc(animation);
  glutMainLoop();
  return 0;
}

void init(void)
{
  glClearColor(0, 0, 0, 0);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
}

void DrawCube(void)
{

  glMatrixMode(GL_MODELVIEW);
  // clear the drawing buffer.
  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, -10.5);
  glRotatef(xRotated, 1.0, 0.0, 0.0);
  // rotation about Y axis
  glRotatef(yRotated, 0.0, 1.0, 0.0);
  // rotation about Z axis
  glRotatef(zRotated, 0.0, 0.0, 1.0);
  glBegin(GL_QUADS);               // Draw The Cube Using quads
  glColor3f(0.0f, 1.0f, 0.0f);     // Color Blue
  glVertex3f(1.0f, 1.0f, -1.0f);   // Top Right Of The Quad (Top)
  glVertex3f(-1.0f, 1.0f, -1.0f);  // Top Left Of The Quad (Top)
  glVertex3f(-1.0f, 1.0f, 1.0f);   // Bottom Left Of The Quad (Top)
  glVertex3f(1.0f, 1.0f, 1.0f);    // Bottom Right Of The Quad (Top)
  glColor3f(1.0f, 0.5f, 0.0f);     // Color Orange
  glVertex3f(1.0f, -1.0f, 1.0f);   // Top Right Of The Quad (Bottom)
  glVertex3f(-1.0f, -1.0f, 1.0f);  // Top Left Of The Quad (Bottom)
  glVertex3f(-1.0f, -1.0f, -1.0f); // Bottom Left Of The Quad (Bottom)
  glVertex3f(1.0f, -1.0f, -1.0f);  // Bottom Right Of The Quad (Bottom)
  glColor3f(1.0f, 0.0f, 0.0f);     // Color Red
  glVertex3f(1.0f, 1.0f, 1.0f);    // Top Right Of The Quad (Front)
  glVertex3f(-1.0f, 1.0f, 1.0f);   // Top Left Of The Quad (Front)
  glVertex3f(-1.0f, -1.0f, 1.0f);  // Bottom Left Of The Quad (Front)
  glVertex3f(1.0f, -1.0f, 1.0f);   // Bottom Right Of The Quad (Front)
  glColor3f(1.0f, 1.0f, 0.0f);     // Color Yellow
  glVertex3f(1.0f, -1.0f, -1.0f);  // Top Right Of The Quad (Back)
  glVertex3f(-1.0f, -1.0f, -1.0f); // Top Left Of The Quad (Back)
  glVertex3f(-1.0f, 1.0f, -1.0f);  // Bottom Left Of The Quad (Back)
  glVertex3f(1.0f, 1.0f, -1.0f);   // Bottom Right Of The Quad (Back)
  glColor3f(0.0f, 0.0f, 1.0f);     // Color Blue
  glVertex3f(-1.0f, 1.0f, 1.0f);   // Top Right Of The Quad (Left)
  glVertex3f(-1.0f, 1.0f, -1.0f);  // Top Left Of The Quad (Left)
  glVertex3f(-1.0f, -1.0f, -1.0f); // Bottom Left Of The Quad (Left)
  glVertex3f(-1.0f, -1.0f, 1.0f);  // Bottom Right Of The Quad (Left)
  glColor3f(1.0f, 0.0f, 1.0f);     // Color Violet
  glVertex3f(1.0f, 1.0f, -1.0f);   // Top Right Of The Quad (Right)
  glVertex3f(1.0f, 1.0f, 1.0f);    // Top Left Of The Quad (Right)
  glVertex3f(1.0f, -1.0f, 1.0f);   // Bottom Left Of The Quad (Right)
  glVertex3f(1.0f, -1.0f, -1.0f);  // Bottom Right Of The Quad (Right)
  glEnd();                         // End Drawing The Cube
  glFlush();
}

void animation(void)
{

  yRotated += 0.01;
  xRotated += 0.02;
  DrawCube();
}

void reshape(int x, int y)
{
  if (y == 0 || x == 0)
    return; //Nothing is visible then, so return
  //Set a new projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //Angle of view:40 degrees
  //Near clipping plane distance: 0.5
  //Far clipping plane distance: 20.0

  gluPerspective(40.0, (GLdouble)x / (GLdouble)y, 0.5, 20.0);
  glMatrixMode(GL_MODELVIEW);
  glViewport(0, 0, x, y); //Use the whole window for rendering
}