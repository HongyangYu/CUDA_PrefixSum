# CUDA_PrefixSum

Cuda prefix sum and weighted prefix sum

Prefix Sum:
Input numbers:	1	2	3	4	5	6	...

Output prefix sums:	1	3	6	10	15	21	...

Weighted Prefix Sum:
INPUT: An integer n and an array A= (a_1,...,a_n) of floating point numbers .

OUTPUT: An array of legth n: (a_1, 2a_1+ a_2, 3a_1+2a_2+a_3, 4a_1+ 3a_2+ 2a_3 + a_4, ...) That is, the ith member of your array must be i*a_1 + (i-1)a_2 + ... + 2a_{i-1} + a_i.

Design a parallel algorithm that runs in TIME(log n) and implement it on the CUDA platform.

Example input: (2,0,7) Example output: (2, 4, 13)
