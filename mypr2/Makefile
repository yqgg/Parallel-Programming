CC = g++
# Compiler flags
CFLAGS := -O3 -g -Wall -Wextra -Wno-unused-parameter 
# Libraries to link
LIBS   := -fopenmp
default: sparsematmult

sparsematmult: sparsematmult.cpp
	${CC} ${CFLAGS} ${LIBS} -o $@ sparsematmult.cpp -std=c++11

clean:
	rm sparsematmult