#!/bin/bash
#
# export THREADS=64
# SBATCH --job-name=pl-lab3
# SBACTH --partition=cpu
# CPU Cores = 2*omp_get_max_threads()
# SBATCH --cpus-per-task=$THREADS
# SBATCH --mem-per-cpu=8G
# SBATCH --nodes=1
# SBATCH --output=pl-lab3-%j.out
# SBATCH --time=10:00
# SBATCH --mail-type=ALL
# SBATCH --mail-user=ygoh2@scu.edu

# export OMP_NUM_THREADS=$THREADS
# export OMP_PLACES=cores
# export OMP_PROC_BIND=true

OUTPUT_DIR="./mat_runs"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# FILE="$OUTPUT_DIR/mat_runs/mat-thread-times-$i.txt" does not work, keeps returning FILE: command not found
make clean && make
./mat_vec_mult.out 1 in_mat.txt in_vec.txt 1000 5000 5000 >> mat_runs/mat-thread-times-0.txt
./mat_vec_mult.out 2 in_mat.txt in_vec.txt 1000 5000 5000 >> mat_runs/mat-thread-times-0.txt
./mat_vec_mult.out 4 in_mat.txt in_vec.txt 1000 5000 5000 >> mat_runs/mat-thread-times-0.txt
./mat_vec_mult.out 8 in_mat.txt in_vec.txt 1000 5000 5000 >> mat_runs/mat-thread-times-0.txt
./mat_vec_mult.out 16 in_mat.txt in_vec.txt 1000 5000 5000 >> mat_runs/mat-thread-times-0.txt
./mat_vec_mult.out 32 in_mat.txt in_vec.txt 1000 5000 5000 >> mat_runs/mat-thread-times-0.txt
./mat_vec_mult.out 64 in_mat.txt in_vec.txt 1000 5000 5000 >> mat_runs/mat-thread-times-0.txt


python generate-plots.py -n mat_runs/mat-thread-times-0.txt

# make clean && make
# ./mat_vec_mult.out > out.txt
# ./mat_vec_mult.out in_mat.txt in_vec.txt 1000 5000 5000 >> out.txt

