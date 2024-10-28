#!/bin/bash
#
# export THREADS=64
#SBATCH --job-name=pl-lab3
#SBATCH --partition=cpu
# CPU Cores = 2*omp_get_max_threads()
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --output=pl-lab3-%j.out
#SBATCH --time=10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ygoh2@scu.edu

# export OMP_NUM_THREADS=$THREADS
#export OMP_PLACES=cores
#export OMP_PROC_BIND=true

OUTPUT_DIR="./runs"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

make clean && make
FILE="$OUTPUT_DIR/out.txt"
# usage: matmult nrows ncols ncols2 tileSize nthreads
./matmult_omp 1000 1000 1000 100 1 >> $FILE
./matmult_omp 1000 1000 1000 100 2 >> $FILE
./matmult_omp 1000 1000 1000 100 4 >> $FILE
./matmult_omp 1000 1000 1000 100 8 >> $FILE
./matmult_omp 1000 1000 1000 100 12 >> $FILE
./matmult_omp 1000 1000 1000 100 14 >> $FILE
./matmult_omp 1000 1000 1000 100 16 >> $FILE
./matmult_omp 1000 1000 1000 100 20 >> $FILE
./matmult_omp 1000 1000 1000 100 24 >> $FILE
./matmult_omp 1000 1000 1000 100 28 >> $FILE


cd runs
cat out.txt