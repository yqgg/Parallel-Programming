#include <vector>
#include "Vector.h"
#include <iostream>
#include <fstream>

class Matrix
{
public:
    std::vector<double> buffer; 
    int n_rows;
    int n_cols;
    Matrix(int rows=100, int cols=100);
    ~Matrix();
};

std::vector<double> operator * (const Matrix &a, const Vector &x)
{
    // Write Code Here
}

// Initialize matrix with random values
Matrix::Matrix(int rows, int cols)
{
    this->n_rows = rows;
    this->n_cols = cols;

    srand(42);
    int n = rows * cols;
    for(int i=0; i<n; ++i){
        float value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        this->buffer.push_back(value);
    }

    // Write to File
    std::ofstream fp("matrix.txt");
    for(auto &i : buffer){
        fp << i << std::endl;
    }
    fp.close();

}

Matrix::~Matrix()
{
}
