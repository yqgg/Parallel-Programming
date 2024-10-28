#!/bin/bash
#
# export THREADS=64
#SBATCH --job-name=pl-mypr2
#SBATCH --partition=cpu
# CPU Cores = 2*omp_get_max_threads()
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --output=pl-mypr2-%j.out
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ygoh2@scu.edu

# export OMP_NUM_THREADS=$THREADS
#export OMP_PLACES=cores
#export OMP_PROC_BIND=true

:<<COMMENT
OUTPUT_DIR1="./runs1"
rm -rf $OUTPUT_DIR1
mkdir -p $OUTPUT_DIR1

OUTPUT_DIR2="./runs2"
rm -rf $OUTPUT_DIR2
mkdir -p $OUTPUT_DIR2

OUTPUT_DIR3="./runs3"
rm -rf $OUTPUT_DIR3
mkdir -p $OUTPUT_DIR3
COMMENT

make clean && make
OUTPUT_DIR1="./size1"
OUTPUT_DIR2="./size2"
OUTPUT_DIR3="./size3"
FILLFACTOR=0.05
NTHREADS=8 # change number of threads according to what user wants to be executed
# Executes program three times each iteration and each with different matrix size, fill factor, and num of threads
# Each iteration of the for loop increases the fill factor, but num of threads remains the same. So I did "sbatch run.sh" for each amount
# of threads that needed to be tested.
for i in {1..4} # for loop range is inclusive.
do
    NEWFILLFACTOR=$(bc -l <<< "$FILLFACTOR*$i")
    FILE1="$OUTPUT_DIR1/out-$NEWFILLFACTOR.txt"
    FILE2="$OUTPUT_DIR2/out-$NEWFILLFACTOR.txt"
    FILE3="$OUTPUT_DIR3/out-$NEWFILLFACTOR.txt"
    # usage <program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]
    ./sparsematmult 10000 10000 10000 $NEWFILLFACTOR -t $NTHREADS >> $FILE1
    ./sparsematmult 20000 5300 50000 $NEWFILLFACTOR -t $NTHREADS >> $FILE2
    ./sparsematmult 9000 35000 5750 $NEWFILLFACTOR -t $NTHREADS >> $FILE3
done