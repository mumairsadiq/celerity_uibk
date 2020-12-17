# Celerity Evlauation
Celerity API (https://celerity.github.io/) is used to implement the algorithms

## New Algorithms

Following are the new implemented algorithms:


(1) Longest common substring     
path --> **examples/longest common substing**


(2) Levenshtein distance (NvPD)

path --> **examples/nvpd**

## Screenshots

Screenshot for sample execution can be found at

**screenshot/**

## Executing NvPD

I used following commands on my local machine to execute the nvpd program

(1) cmake ./ -DCMAKE_PREFIX_PATH="/usr/local/lib" -DHIPSYCL_PLATFORM=cuda -DHIPSYCL_GPU_ARCH=sm_30 -DCMAKE_BUILD_TYPE=Release

(2) make

(3) mpirun -n 4 ./nvpd q1.txt 100 q2.txt 100

## Executing LCSubstr

I used following commands on my local machine to execute the LCSubstr program

(1) cmake ./ -DCMAKE_PREFIX_PATH="/usr/local/lib" -DHIPSYCL_PLATFORM=cuda -DHIPSYCL_GPU_ARCH=sm_30 -DCMAKE_BUILD_TYPE=Release

(2) make

(3) mpirun -n 4 ./LCSubstr q1.txt 100 q2.txt 100

