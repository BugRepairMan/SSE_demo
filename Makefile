CC = g++
FLAGS = -std=c++11 -mavx2 -O2 -fopenmp

all: test

test: test.cpp
	$(CC) $(FLAGS) $< -o $@

clean:
	rm test
