CC = g++
FLAGS = -std=c++11 -mavx2 -O2
LIBS := -framework OpenCL

all: test

test: main.cpp cl_func.cpp kernel.cl
	$(CC) $(FLAGS) $< -o $@ $(LIBS)

clean:
	rm test
