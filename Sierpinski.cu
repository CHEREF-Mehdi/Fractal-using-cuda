#include <GL/glut.h>
#include <cstdlib>
#include <ctime>
#include <armadillo>
#include <cuda.h>
#include <stack>

//files.associations

using namespace std;

// menu item
#define MENU_SMOOTH 1
#define MENU_FLAT 0

struct fractalLevel
{
    vector<arma::Mat<GLfloat>> tList;
    arma::Mat<GLfloat> trgl;
    int iteration;
};

struct transformLevel
{
    arma::Mat<GLfloat> transf = {{1.0, 0.0, 0.0},
                                 {0.0, 1.0, 0.0},
                                 {0.0, 0.0, 1.0}};
    vector<int> trans;
    int iteration;
};

arma::Mat<GLfloat> Triangle = {{0.0, 0.0, 0.0},
                               {1.0, 0.0, 0.0},
                               {0.0, 1.0, 0.0}};

arma::Mat<GLfloat> Triangle2 = {{1.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0}};

vector<arma::Mat<GLfloat>> transfMatList{
    {{1.0, 0.5, 0.5},
     {0.0, 0.5, 0.0},
     {0.0, 0.0, 0.5}},
    {{0.5, 0.0, 0.0},
     {0.5, 1.0, 0.5},
     {0.0, 0.0, 0.5}},
    {{0.5, 0.0, 0.0},
     {0.0, 0.5, 0.0},
     {0.5, 0.5, 1.0}}};

double colors[1000] = {0};

int iterations = 0, maxIteration = 11;
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

void divideTriangle(arma::Mat<GLfloat> trgl, vector<arma::Mat<GLfloat>> TransfList, int iteration)
{
    if (iteration == 0)
    {
        trgl = trgl.t();
        int n_rows = trgl.n_rows;
        GLfloat **poly = toGLfloatPoints(trgl, n_rows);
        drawPolygone(poly, n_rows);
    }
    else
    {
        divideTriangle(TransfList[0] * trgl, TransfList, iteration - 1);
        divideTriangle(TransfList[1] * trgl, TransfList, iteration - 1);
        divideTriangle(TransfList[2] * trgl, TransfList, iteration - 1);
    }
}

void divideTriangleIterative(arma::Mat<GLfloat> trgl, vector<arma::Mat<GLfloat>> TransfList, int iter)
{
    stack<fractalLevel> stk;
    fractalLevel level;
    while (true)
    {
        while (iter > 0)
        {
            iter--;
            level.iteration = iter;
            //level.tList = TransfList;
            level.trgl = TransfList[2] * trgl;
            stk.push(level);
            level.trgl = TransfList[1] * trgl;
            stk.push(level);
            trgl = TransfList[0] * trgl;
        }

        trgl = trgl.t();
        int n_rows = trgl.n_rows;
        GLfloat **poly = toGLfloatPoints(trgl, n_rows);
        drawPolygone(poly, n_rows);

        if (stk.empty())
            break;
        else
        {
            level = stk.top();
            stk.pop();
            trgl = level.trgl;
            iter = level.iteration;
            TransfList = level.tList;
        }
    }
}

void getListTransform(arma::Mat<GLfloat> trgl, vector<arma::Mat<GLfloat>> TransfList, int iter)
{
    stack<transformLevel> stk;
    transformLevel level;
    arma::Mat<GLfloat> t, tr = level.transf;

    while (true)
    {
        transformLevel l;
        while (iter > 0)
        {
            iter--;
            level.iteration = iter;
            l = level;
            l.trans.push_back(2);
            stk.push(l);
            l = level;
            l.trans.push_back(1);
            stk.push(l);
            l = level;
            l.trans.push_back(1);
        }
        for (auto i = l.trans.cbegin(); i != l.trans.cend(); ++i)
        {
        }
        t = level.transf * trgl;
        t = t.t();
        int n_rows = t.n_rows;
        GLfloat **poly = toGLfloatPoints(t, n_rows);
        drawPolygone(poly, n_rows);

        if (stk.empty())
            break;
        else
        {
            level = stk.top();
            stk.pop();
            iter = level.iteration;
            tr = level.transf;
        }
    }
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(zoom, zoom, zoom);
    random(true);
    //divideTriangle(Triangle2, transfMatList, iterations);
    //divideTriangleIterative(Triangle2, transfMatList, iterations);
    getListTransform(Triangle2, transfMatList, iterations);
    glFlush();
}

int main(int argc, char **argv)
{

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
