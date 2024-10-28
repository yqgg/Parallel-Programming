   In program 2, we are writing a program to multiply two sparse matrices. The sparse matrices are stored in a Compressed Sparse Row (CSR) 
data structure. CSR improves the efficiency of the computation by formatting the matrices in a way that removes all zeroes. It is 
unnecessary to include zeroes in matrix-matrix multiplication. More detailed explanation of CSR included below.
   The "run.sh" file helps me execute the "sparsematmult.cpp" program with different parameters that modify matrix size, determine fill 
factor, number of parallel threads, and which file to save the output to. The getSpeedUp.cpp program reads a file and writes the speed up 
value to "output_speedup.txt" so that I can more easily copy all speed up values to excel.

Folder "size 1" refers to matrix size A(10000x10000) and B(10000x10000)
       "size 2" refers to matrix size A(20000x5300) and B(5300x50000)
       "size 3" refers to matrix size A(9000x35000) and B(35000x5750)
The number in the text file name of each "size #" folder is the fill factor under which the execution was run.



CSR Explanation:
In a Compressed Sparse Row (CSR) data structure, you can determine the size of each array (values, indices, and pointer) using the following steps:

Values Array Size: The size of the values array is determined by the total number of non-zero elements in the sparse matrix. This count is typically equal to the size of the values array.

Indices Array Size: The size of the indices array is also equal to the total number of non-zero elements. Each element in the indices array corresponds to a column index of a non-zero element in the matrix.

Pointer Array Size: The size of the pointer array is determined by the number of rows in the matrix plus one. There is an additional element in the pointer array to mark the end of the last row.

Here's a more detailed breakdown of these steps:

Values Array Size:
   - To find the size of the values array, iterate through the non-zero elements of the matrix and increment a counter for each non-zero element encountered.
   - The counter's final value is the size of the values array.

Indices Array Size:
   - The size of the indices array is the same as the size of the values array since each value in the values array corresponds to a column index.
   - You can use the same counter used to count non-zero values for the indices array size.

Pointer Array Size:
   - The size of the pointer array is determined by the number of rows in the matrix plus one. The last element in the pointer array is used to mark the end of the last row.

After determining the sizes of the values, indices, and pointer arrays using these steps, you can allocate memory for these arrays with the correct sizes and then proceed to fill the CSR data structure.

In practice, you would use this size information to allocate memory for the arrays dynamically, ensuring that you have enough space to store all the data. Once the arrays are correctly sized and allocated, you can efficiently represent the sparse matrix in CSR format and perform various operations on it.