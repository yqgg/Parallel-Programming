#include <stdio.h>

// Usage: gdb_tut 1 2 3 4 5 6
// Return: sum of integers passed as cmd line arguments

int main(int argc, char **argv){

    float sum = 0;

    for(int i = 0; i < argc; i++){
        sum += (float)(*argv[i]);
    }

    printf("Sum: %f\n", sum);

    return 0;
}