#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
//#include "fileio.h"

// Usage: sum.out n_threads n_elements
// Return: sum of `n_elements' floating-point numbers passed as cmd line arguments
// Sample Output
// > n_threads time_in_microseconds speed_up

double f(long long x){
  return sin(exp(sqrt(double(x))));
}

int main(int argc, char **argv)
{
  //generate_input();

  int num_threads = atoi(argv[1]);
  long long ix = atoi(argv[2]);
  omp_set_num_threads(num_threads);

  double ts=0;
  double tp=0;

  #pragma omp parallel
  {
    double sum = 0;
    static double start = omp_get_wtime();

    #pragma omp for
    for (long long i = 0; i < ix; i++)
    {
      sum += f(i);
    }
    
    #pragma omp single
    {
      printf("%d", omp_get_num_threads());
      static double end = omp_get_wtime();
      static double time_parallel = (end - start) * 1e6; // in microseconds
      printf(" %.2f", time_parallel);
      tp = time_parallel;  
    }

  }
  
  // Serial Region
  {
    omp_set_num_threads(1);
    double sum = 0;
    static double start = omp_get_wtime();
    for (long long i = 0; i < ix; i++)
    {
      sum += f(i);
    }
    static double end = omp_get_wtime();
    double time_serial = (end - start) * 1e6; // in microseconds
    ts = time_serial;
  }

  // Speed-up := time_serial / time_parallel
  printf(" %.2f\n", ts / tp);
  return 0;
}
