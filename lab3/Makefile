# Compiler
CC = gcc
# Compiler flags
CFLAGS = -O3 -g
# Libraries to link
LIBS = -fopenmp
# Object files
OBJ = mat_vec_mult.o
#Executable name
TARGET = mat_vec_mult.out

all: $(TARGET)
	mv *.o .objs/

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(LIBS) -o $@ $^

%.o : %.c
	$(CC) $(CFLAGS) $(LIBS) -c -o $@ $<

clean:
	rm .objs/*.o
	rm $(TARGET)