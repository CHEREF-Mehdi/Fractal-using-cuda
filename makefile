

NVCC = nvcc

LFLAGS = -arch=sm_35 -rdc=true -std=c++11 -m64

#armadilo
LIBARMADILO= -DARMA_DONT_USE_WRAPPER -lopenblas -llapack

#opengl
LIBOPENGL= -lGL -lGLEW -lGLU -lglut -lm

EXEC= DFS2
SRC= DFS2.cu 

# EXEC= Sierpinski
# SRC= Sierpinski.cu 


$(EXEC): $(SRC)
	$(NVCC) $(LFLAGS) -o $(EXEC) $(SRC) ${LIBOPENGL} ${LIBARMADILO}

all: 	 
	$(EXEC)

clean:
	rm -rf *.o
	rm -rf $(EXEC)