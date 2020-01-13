
#CUDAPATH = /usr/local/cuda
#CFLAGS = -c -m64 -I$(CUDAPATH)/include
#NVCCFLAGS = -c -I$(CUDAPATH)/include

NVCC = nvcc

LFLAGS = -std=c++11 -m64

#armadilo
LIBARMADILO= -DARMA_DONT_USE_WRAPPER -lopenblas -llapack

#opengl
LIBOPENGL= -lGL -lGLEW -lGLU -lglut -lm

EXEC= Sierpinski
SRC= Sierpinski.cu 


Sierpinski: Sierpinski.cu
	$(NVCC) $(LFLAGS) -o $(EXEC) $(SRC) ${LIBOPENGL} ${LIBARMADILO}

all: 	 
	$(EXEC)

clean:
	rm -rf *.o
	rm -rf $(EXEC)