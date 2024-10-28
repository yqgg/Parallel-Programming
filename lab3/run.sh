#!/bin/bash

OUTPUT_DIR="./runs"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

NUMS=1234567890

for i in {0..3}
do
FILE="$OUTPUT_DIR/thread-times-$i.txt"
./sum.out 1 $NUMS >> $FILE 
./sum.out 2 $NUMS >> $FILE
./sum.out 4 $NUMS >> $FILE
./sum.out 8 $NUMS >> $FILE
./sum.out 16 $NUMS >> $FILE
./sum.out 32 $NUMS >> $FILE
./sum.out 64 $NUMS >> $FILE
done

python generate-plots.py -n runs/thread-times-0.txt