/* assert */
#include <assert.h>

/* errno */
#include <errno.h>

/* fopen, fscanf, fprintf, fclose */
#include <stdio.h>

/* EXIT_SUCCESS, EXIT_FAILURE, malloc, free */
#include <stdlib.h>
#include <omp.h>
#include <time.h>


static int create_mat(size_t const nrows, size_t const ncols, double ** const matp)
{
    double * mat=NULL;
    if (!(mat = (double*) malloc(nrows*ncols*sizeof(*mat)))) {
        goto cleanup;
    }

    /** Initialize matrix with random values **/
    for(size_t i = 0; i < nrows; i++){
        for (size_t j = 0; j < ncols; j++){
            mat[(i * ncols) + j] = (double)(rand() % 1000) / 353.0;
        }
    }
    /** End random initialization **/

    *matp = mat;

    return 0;

    cleanup:
    free(mat);
    return -1;
}

static int mult_mat(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
    size_t i, j, k;
    double sum;
    double * C = NULL;

    if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
        goto cleanup;
    }

    omp_set_num_threads(1);
    #pragma omp parallel
    {
        static double start;
        start = omp_get_wtime();
        for (i=0; i<n; ++i) {
            for (j=0; j<p; ++j) {
            for (k=0, sum=0.0; k<m; ++k) {
                sum += A[i*m+k] * B[k*p+j];
            }
            C[i*p+j] = sum;
            }
        }
        #pragma omp single
        {
            static double end;
            end = omp_get_wtime();
            printf("Serial execution time (%d thread): %.4f seconds\n", omp_get_num_threads(), end-start);
        }
    }

    *Cp = C;

    return 0;

    cleanup:
    free(C);

    /*failure:*/
    return -1;
}

static int mult_mat_no_tiling(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp, size_t nthreads)
{
    size_t i, j, k;
    double sum;
    double * C = NULL;

    if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
        goto cleanup;
    }

    omp_set_num_threads(nthreads);
    #pragma omp parallel shared(C) private(i, j, k, sum)
    {
        static double start;
        start = omp_get_wtime();
        #pragma omp for
        for (i=0; i<n; ++i) {
            for (j=0; j<p; ++j) {
                for (k=0, sum=0.0; k<m; ++k) {
                    sum += A[i*m+k] * B[k*p+j];
                }
            C[i*p+j] = sum;
            }
        }
        #pragma omp single
        {
            static double end;
            end = omp_get_wtime();
            printf("No tiling parallel execution time (%d threads): %.4f seconds\n", omp_get_num_threads(), end-start);
        }
    }

    *Cp = C;

    return 0;

    cleanup:
    free(C);

    /*failure:*/
    return -1;
}

static int mult_mat_tiling(size_t const n, size_t const m, size_t const p,
                            double const * const A, double const * const B,
                            double ** const Cp, size_t tileSize, size_t nthreads) 
{
    /*  A: n x m 
        B: m x p
        C: n x p */

    size_t i, j, k, I, J, K; //'s' for start and 'e' for end (Ns = Nstart, Ne = Nend)
    double sum;
    double * C = NULL;

    if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
        goto cleanup;
    }

    omp_set_num_threads(nthreads);
    #pragma omp parallel shared(C) private(i, j, k, I, J, K, sum)
    {
        static double start;
        start = omp_get_wtime();
        #pragma omp for collapse(3)
        for(i = 0; i<n; i+=tileSize) {
            for(j = 0; j<p; j+=tileSize) {
                for(k = 0; k<m; k+=tileSize) {
                    for(I = i; I< i+tileSize; I++) {
                        for(J = j; J< j+tileSize; J++) {
                            for(K = k, sum = 0; K< k+tileSize; K++) {
                                sum += A[I*m+k] * B[K*p+j];
                            }
                            C[I*p+J] += sum;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            static double end;
            end = omp_get_wtime();
            printf("Tiling parallel execution time (%lu tiles, %d threads): %.4f seconds\n\n", tileSize,  omp_get_num_threads(), end-start);
        }
    }

    *Cp = C;

    return 0;

    cleanup:
    free(C);

    /*failure:*/
    return -1;
}

int main(int argc, char * argv[])
{
    // size_t stored an unsigned integer
    size_t nrows, ncols, ncols2, tileSize, nthreads;
    double * A=NULL, * B=NULL, * C=NULL;

    if (argc != 6) {
        fprintf(stderr, "usage: matmult nrows ncols ncols2 tileSize nthreads\n");
        goto failure;
    }

    nrows = atoi(argv[1]);
    ncols = atoi(argv[2]);
    ncols2 = atoi(argv[3]);
    tileSize = atoi(argv[4]);
    nthreads = atoi(argv[5]);

    printf("NUM OF THREADS: %lu\n", nthreads);
    printf("MATRIX SIZE: A[%lu][%lu], ", nrows, ncols);
    printf("B[%lu][%lu], ", ncols, ncols2);
    printf("C[%lu][%lu]\n", nrows, ncols2);

    if (create_mat(nrows, ncols, &A)) {
        perror("error");
        goto failure;
    }

    if (create_mat(ncols, ncols2, &B)) {
        perror("error");
        goto failure;
    }

    if (mult_mat(nrows, ncols, ncols2, A, B, &C)) {
        perror("error");
        goto failure;
    }

    if(mult_mat_no_tiling(nrows, ncols, ncols2, A, B, &C, nthreads)) {
        perror("error");
        goto failure;
    }

    if(mult_mat_tiling(nrows, ncols, ncols2, A, B, &C, tileSize, nthreads)) {
        perror("error");
        goto failure;
    }

    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;

    failure:
    if(A){
        free(A);
    }
    if(B){
        free(B);
    }
    if(C){
        free(C);
    }
    return EXIT_FAILURE;
}
