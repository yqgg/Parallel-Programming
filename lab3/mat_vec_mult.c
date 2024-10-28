#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
//#include <string>

// Usage: ./mat_vec_mult.out {in_mat.txt | in_vec.txt | nrow | ncol1 | ncol2} > out.txt
// Return: Computes the matrix vector product

void writeToFile(const char *filename) {
    FILE *file = fopen(filename, "w");
    if(file == NULL) {
        perror("Failed to open the file");
        return;
    }
    fclose(file);
}

typedef struct {
    double *buffer;
    int n_rows;
    int n_cols;
} Matrix;

typedef struct {
    double *buffer;
    int size;
} Vector;

Matrix createMatrix(int rows, int cols) {
    Matrix mat;
    mat.n_rows = rows;
    mat.n_cols = cols;
    int n = rows * cols;
    mat.buffer = (double *)malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        double value = (double)rand() / RAND_MAX;
        mat.buffer[i] = value;
    }

    // Write to File
    writeToFile("1D_mat.txt");
    return mat;
}

void destroyMatrix(Matrix *mat) {
    free(mat->buffer);
    mat->n_rows = 0;
    mat->n_cols = 0;
}

Vector createVector(int size) {
    Vector vec;
    vec.size = size;
    vec.buffer = (double *)malloc(sizeof(double) * size);
    // Initialize the vector with some values
    srand(1337);
    for (int i = 0; i < vec.size; i++) {
        double value = (double)rand() / RAND_MAX;
        vec.buffer[i] = value;
    }
    return vec;
}

void destroyVector(Vector *vec) {
    free(vec->buffer);
    vec->size = 0;
}


void matrixVectorMultiply(const Matrix *mat, const Vector *vec, double num_threads) {
    /* os = one-dimensional series
       op = one-dimensional parallel*/
    double os, op = 0;

    //1D parallel region: using matrix[i] instead of matrix[i][j]
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        Vector result = createVector(mat->n_rows);
        static double start;
        start = omp_get_wtime();
        #pragma omp for
        for (int i = 0; i < mat->n_rows; i++) {
            result.buffer[i] = 0;
            for (int j = 0; j < mat->n_cols; j++) {
                result.buffer[i] += mat->buffer[i * mat->n_cols + j] * vec->buffer[j];
            }
        }

        #pragma omp single
        {
            printf("%d", omp_get_num_threads());
            static double end;
            end = omp_get_wtime();
            static double time_parallel; 
            time_parallel = (end - start) * 1e6; // in microseconds
            printf(" %.2f", time_parallel);
            op = time_parallel;  
        } 
    }

    //1D serial region
    {
        omp_set_num_threads(1);
        Vector result = createVector(mat->n_rows);
        static double start;
        start = omp_get_wtime();
        for (int i = 0; i < mat->n_rows; i++) {
            result.buffer[i] = 0;
            for (int j = 0; j < mat->n_cols; j++) {
                result.buffer[i] += mat->buffer[i * mat->n_cols + j] * vec->buffer[j];
            }
        }
        static double end;
        end = omp_get_wtime();
        static double time_serial;
        time_serial = (end-start) * 1e6; //in microseconds
        os = time_serial;
    }
    printf(" %.2f\n", os / op);
}

int main(int argc, char **argv){
    int nrows = atoi(argv[4]);
    int ncol1 = atoi(argv[5]);
    int ncol2 = atoi(argv[6]);
    
    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);

    /* ts = two-dimensional series
       tp = two-dimensional parallel*/
    double ts, tp = 0;

    // Allocate memory for the matrix and vector
    double **matrix = (double **)malloc(nrows * sizeof(double *));
    for (int i = 0; i < nrows; i++) {
        matrix[i] = (double *)malloc(ncol1 * sizeof(double));
    }
    double *vector = (double *)malloc(ncol2 * sizeof(double));
    
    //generate matrix with random values
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncol1; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }

    //generate vector with random values
    for (int i = 0; i < ncol2; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }

    //write matrix to .txt
    FILE *file = fopen(argv[1], "w");
    if (file == NULL) {
        perror("Error opening file for writing");
        exit(1);
    }
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncol1; j++) {
            fprintf(file, "%lf\n", matrix[i][j]);
        }
    }
    fclose(file);

    //write vector to .txt
    FILE *file2 = fopen(argv[2], "w");
    if (file2 == NULL) {
        perror("Error opening file for writing");
        exit(1);
    }
    for (int i = 0; i < ncol2; i++) {
        fprintf(file2, "%lf\n", vector[i]);
    }
    fclose(file2);

    //parallel region 2D: multiply matrix and vector using row col index (mat[i][j]), store execution as local variable tp, and print execution time to .out file
    #pragma omp parallel
    {
        double *result_parallel = (double *)malloc(ncol1 * sizeof(double));
        static double start;
        start = omp_get_wtime();
        
        #pragma omp for
        for(int i = 0; i < nrows; i++) {
            //result_parallel[i] = 0;
            for(int j = 0; j < ncol1; j++) {
                result_parallel[i] += matrix[i][j] * vector[j];
                //printf("%f\n", result[i]);
            }
        }

        #pragma omp single
        {
            printf("%d", omp_get_num_threads());
            static double end;
            end = omp_get_wtime();
            static double time_parallel; 
            time_parallel = (end - start) * 1e6; // in microseconds
            printf(" %.2f", time_parallel);
            tp = time_parallel;  
        }  
    }

    //serial region 2D: multiply mat and vect using row col index (mat[i][j]) and store execution time as local variable ts
    {
        omp_set_num_threads(1);
        double *result_series = (double *)malloc(ncol1 * sizeof(double));
        static double start;
        start = omp_get_wtime();
        for(int i = 0; i < nrows; i++) {
            //result_series[i] = 0;
            for(int j = 0; j < ncol1; j++) {
                result_series[i] += matrix[i][j] * vector[j];
                //printf("%f\n", result[i]);
            }
        }
        static double end;
        end = omp_get_wtime();
        static double time_serial;
        time_serial = (end-start) * 1e6;
        ts = time_serial;
        //printf("time-series (2D) = %f\n", (end - start) * 1e6);
    }

    // Speed-up := time_serial / time_parallel
    printf(" %.2f\n", ts / tp);



//1D buffer method below


    //from lab1: result.at(i) += a.buffer.at(i * a.n_cols + j) * x.buffer[j];
    Matrix matrix_1D = createMatrix(nrows, ncol1);
    Vector vector_1D = createVector(ncol2);

    matrixVectorMultiply(&matrix_1D, &vector_1D, num_threads);

    destroyMatrix(&matrix_1D);
    destroyVector(&vector_1D);

    return 0;
}
