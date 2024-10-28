#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>
#include <iomanip>

using namespace std;

using idx_t = std::uint32_t; 
using val_t = float; 
using ptr_t = std::uintptr_t;

/**
 * CSR (Compressed Sparse Row) structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of cols
  idx_t * ind; // column ids, stores column indices corresponding to the non-zero values, size determined by num of non-zero elements 
  val_t * val; // values, stores non-zero values of matrix in row-major order
  ptr_t * ptr; // pointers (start of row in ind/val), indicates starting index of each row in the 'val' and 'ind' arrays

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(nnz > ptr[this->nrows]){
      if(ind){
        ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
      } else {
        ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
      }
      if(!ind){
        throw std::runtime_error("Could not allocate ind array.");
      }
      if(val){
        val = (val_t*) realloc(val, sizeof(val_t) * nnz);
      } else {
        val = (val_t*) malloc(sizeof(val_t) * nnz);
      }
      if(!val){
        throw std::runtime_error("Could not allocate val array.");
      }
    }
    this->nrows = nrows;
  }

  /** 
   * Transpose matrix.
   * Need to transpose because the num of cols in matrix A may not match the num of 
   * rows in matrix B. Num of cols in A must equal num of rows in B for the operation
   * to be defined.
   */ 
  void transpose()
  {
    /* TODO: Implement matrix transposition */
      csr_t *result = new csr_t();
      result->ncols = this->nrows;
      result->reserve(this->ncols, this->ptr[this->nrows]);

      #pragma omp parallel for
      for (idx_t i = 0; i < result->nrows; i++) {
          result->ptr[i] = 0;
      }

      #pragma omp parallel for
      for (idx_t i = 0; i < this->nrows; i++) {
          for (idx_t j = this->ptr[i]; j < this->ptr[i + 1]; j++) {
              result->ptr[this->ind[j] + 1]++;
          }
      }

      for (idx_t i = 1; i < result->nrows; i++) {
          result->ptr[i] += result->ptr[i - 1];
      }

      for (idx_t i = 0; i < this->nrows; i++) {
          for (idx_t j = this->ptr[i]; j < this->ptr[i + 1]; j++) {
              int idx = result->ptr[this->ind[j]]++;
              result->val[idx] = this->val[j];
              result->ind[idx] = i;
          }
      }

      for (idx_t i = result->nrows; i > 0; i--) {
          result->ptr[i] = result->ptr[i - 1];
      }
      result->ptr[0] = 0;

      // Copy the transposed data back to the current matrix
      this->ncols = result->ncols;
      this->reserve(result->nrows, result->ptr[result->nrows]);
      memcpy(this->ind, result->ind, result->ptr[result->nrows] * sizeof(idx_t));
      memcpy(this->val, result->val, result->ptr[result->nrows] * sizeof(val_t));
      memcpy(this->ptr, result->ptr, (result->nrows + 1) * sizeof(ptr_t));

      delete result;
  }

  

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }
    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  //my function to print values of the matrices to check if math is right
  void printValues() const
  {
      for (idx_t i = 0; i < nrows; i++) {
          for (ptr_t j = ptr[i]; j < ptr[i + 1]; j++) {
              cout << "Row: " << i << ", Col: " << ind[j] << ", Value: " << val[j] << endl;
          }
      }
  }


  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat){
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}

/**
 * Multiply A and B and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B The transpose of matrix B.
 * @param C  Output matrix
 */
void omp_sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
    //auto B = transpose(B);
    //test_matrix(B);
    //C->nrows = A->nrows; //if uncommented, it causes a seg fault
    //C->ncols = B->ncols;

    //initialize matrix C and set number of nonzero elements to reserve space
    C->reserve(A->nrows, 1);
    C->ptr[A->nrows] = 0;
    //total size of matrix C
    int totalSize = 0; 
    //initialize an array of pointers of size C->nrows pointing to rows of output C
    csr_t ** C_output_rows = new csr_t *[C->nrows]; 
    
    //paralle region
    #pragma omp parallel
    {
      #pragma omp for
      for(idx_t i = 0; i < A->nrows; i++) {
        //initialize new row for matrix C
        C_output_rows[i] = new csr_t(); 
        //Initialize accumulator array for the current row, {} is a feature of c++11 that initializes an array of all zeroes
        val_t *accumulator = new val_t[B->ncols]{};  
        std::vector<idx_t> indexCount; //columns/indices of non-zero values of C
        //A*B = C for current row of A. Traverse matrix A's 'ptr' array to locate the first non-zero element of each row of A
        for(idx_t j = A->ptr[i]; j < A->ptr[i+1]; j++) {
          int aCol = A->ind[j]; //int or idx_t?
          val_t value = A->val[j];
          //Do I need to transpose B so that the 'ptr' array points to the columns rather than the rows? Answer: no, col of A and row of B already match
          //iterate over nonzero elements of the corresponding col in matrix B
          for(idx_t k = B->ptr[aCol]; k <  B->ptr[aCol+1]; k++ ) {
            if(accumulator[B->ind[k]] == 0) { 
              indexCount.push_back(B->ind[k]);
            }
            //multiply and accumulate result in accumulator array
            accumulator[B->ind[k]] += (value * B->val[k]); 
          }
        }    

        //set properties of output row in matrix C
        C_output_rows[i]->ncols = C->ncols;
        ptr_t size = indexCount.size();
        C_output_rows[i]->reserve(1, size);
        C->ptr[i+1] = size;

        #pragma omp critical
        totalSize += size;

        idx_t index = 0;
        for(idx_t j = 0; j<size; j++) {
            int accIndex = indexCount[j];
            C_output_rows[i]->val[index] = accumulator[accIndex];
            C_output_rows[i]->ind[index] = accIndex;
            index++;
        }
        free(accumulator); 
      }

      /* Used to verify number of threads in parallel region
      #pragma omp single
      {
        cout << "num threads in omp_sparsemult: " << omp_get_num_threads() << endl;
      } */
    }

    C->reserve(C->nrows, totalSize);
    for (idx_t i = 0; i < C->nrows; i++) {
        memcpy(&C->val[C->ptr[i]], C_output_rows[i]->val, C->ptr[i + 1] * sizeof(val_t));
        memcpy(&C->ind[C->ptr[i]], C_output_rows[i]->ind, C->ptr[i + 1] * sizeof(idx_t));
        C->ptr[i+1] += C->ptr[i];
    }
}

void sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
    //auto B = transpose(B);
    //test_matrix(B);
    //C->nrows = A->nrows; //if uncommented, it causes a seg fault
    //C->ncols = B->ncols;
    omp_set_num_threads(1);
    C->reserve(A->nrows, 1);
    C->ptr[A->nrows] = 0; //shouldn't it be first point is 0
    int totalSize = 0; //total size of matrix C

    csr_t ** C_output_rows = new csr_t *[C->nrows]; //an array of pointers of size C->nrows pointing to rows of output C
    for(idx_t i = 0; i < A->nrows; i++) {
      C_output_rows[i] = new csr_t();
      val_t *accumulator = new val_t[B->ncols]{};  
      //A*B = C
      std::vector<idx_t> indexCount; //columns/indices of non-zero values of C
      for(idx_t j = A->ptr[i]; j < A->ptr[i+1]; j++) { //traverse matrix A's 'ptr' array to locate the first non-zero element of each row of A
        int aCol = A->ind[j]; //int or idx_t?
        val_t value = A->val[j];
        //Do I need to transpose B so that the 'ptr' array points to the columns rather than the rows?
        for(idx_t k = B->ptr[aCol]; k <  B->ptr[aCol+1]; k++ ) {
          if(accumulator[B->ind[k]] == 0) { 
            indexCount.push_back(B->ind[k]);
          }
          accumulator[B->ind[k]] += (value * B->val[k]); //multiply 
        }
      }

      C_output_rows[i]->ncols = C->ncols;
      ptr_t size = indexCount.size();
      C_output_rows[i]->reserve(1, size);
      C->ptr[i+1] = size;

      totalSize += size;

      idx_t index = 0;
      for(idx_t j = 0; j<size; j++) {
          int accIndex = indexCount[j];
          C_output_rows[i]->val[index] = accumulator[accIndex];
          C_output_rows[i]->ind[index] = accIndex;
          index++;
      }
      free(accumulator);
    }

    C->reserve(C->nrows, totalSize);
    for (idx_t i = 0; i < C->nrows; i++) {
        memcpy(&C->val[C->ptr[i]], C_output_rows[i]->val, C->ptr[i + 1] * sizeof(val_t));
        memcpy(&C->ind[C->ptr[i]], C_output_rows[i]->ind, C->ptr[i + 1] * sizeof(idx_t));
        C->ptr[i+1] += C->ptr[i];
    }
}


int main(int argc, char *argv[])
{
  if(argc < 4){
    cerr << "Invalid options." << endl << "<program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]" << endl;
    exit(1);
  }
  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int ncols2 = atoi(argv[3]);
  double factor = atof(argv[4]);
  int nthreads = 1;
  if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
    nthreads = atoi(argv[6]);
    omp_set_num_threads(nthreads);
  }
  cout << "A_nrows: " << nrows << endl;
  cout << "A_ncols: " << ncols << endl;
  cout << "B_nrows: " << ncols << endl;
  cout << "B_ncols: " << ncols2 << endl;
  cout << "factor: " << factor << endl;
  cout << "nthreads: " << nthreads << endl;

  /* initialize random seed: */
  srand (time(NULL));

  auto A = csr_t::random(nrows, ncols, factor);
  auto B = csr_t::random(ncols, ncols2, factor); // Note B is not transposed.
  test_matrix(A);
  test_matrix(B);
  auto C = new csr_t(); // Note that C has no data allocations so far.

  //parallel execution
  cout << A->info("A") << endl;
  //A->printValues();
  cout << B->info("B") << endl;
  //B->printValues();
  auto t1 = omp_get_wtime();
  /* Optionally transpose matrix B (Must implement) */
  //B->transpose();
  omp_sparsematmult(A, B, C);
  auto t2 = omp_get_wtime();
  cout << C->info("C") << endl;
  //C->printValues();
  cout << setprecision(4) << fixed;
  float parallel_time = (t2-t1)* 1e6;
  cout << "Parallel execution time: " << parallel_time << " microseconds" << endl;


  //serial execution
  auto T1 = omp_get_wtime();
  /* Optionally transpose matrix B (Must implement) */
  //B->transpose();
  sparsematmult(A, B, C);
  auto T2 = omp_get_wtime();
  cout << C->info("C") << endl;
  //C->printValues();
  cout << setprecision(4) << fixed;
  float serial_time = (T2-T1)* 1e6;
  cout << "Serial execution time: " << serial_time << " microseconds" << endl;
  cout << "Speed up: " << serial_time/parallel_time << "\n" << endl;

  delete A;
  delete B;
  delete C;

  return 0;
}
