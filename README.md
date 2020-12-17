# Celerity Evlauation
Celerity API (https://celerity.github.io/) to implement mentioned algorithms

## New Algorithms

Following are the new implemented algorithms that are implemented


Longest common substring     
path --> examples/longest common substing


Levenshtein distance (NvPD)

path --> examples/nvpd

## Screenshots

Screenshot for sample execution can be found at

screenshot/

## Executing NvPD

I used following commands on my local machine to execute the nvpd program

cmake ./ -DCMAKE_PREFIX_PATH="/usr/local/lib" -DHIPSYCL_PLATFORM=cuda -DHIPSYCL_GPU_ARCH=sm_30 -DCMAKE_BUILD_TYPE=Release

make

mpirun -n 4 ./nvpd q1.txt 100 q2.txt 100

## Executing LCSubstr

I used following commands on my local machine to execute the nvpd program

cmake ./ -DCMAKE_PREFIX_PATH="/usr/local/lib" -DHIPSYCL_PLATFORM=cuda -DHIPSYCL_GPU_ARCH=sm_30 -DCMAKE_BUILD_TYPE=Release

make

mpirun -n 4 ./LCSubstr q1.txt 100 q2.txt 100

