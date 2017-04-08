//
//  main.c
//  matrix_multiplication_cuda
//
//  Created by Meng Li on 2/21/17.
//  Copyright © 2017 Meng Li. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define BLOCK_SIZE 32

/*******************************************************************************/
//A naive implementation of matrix multiplication on cuda
//d_a : m by n input matrix
//d_b : n by m input matrix
//d_c : m by m output matrix

__global__ void matrix_multiply_cuda(int* d_a, int* d_b, int* d_c, int m, int n) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;    // Row i of matrix C
    int j = blockIdx.x * blockDim.x + threadIdx.x;    // Column j of matrix C
    
    //Compute c[i][j] = a[i][k]+b[k][j] over k = 0...n-1
    int cell = 0;
    for (int k=0; k<n; k++)
        cell += d_a[i*n+k] * d_b[k*m+j];
    d_c[i*m+j]=cell;
}

__global__ void matrix_multiply_tiling_cuda(int* A, int* B, int* C, int m, int n) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = n * blockDim.y * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + n - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = blockDim.x;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = blockDim.x * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = blockDim.y * m;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    int Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
	// Suppose to be As[blockDim.y][blockDim.x] but need dynamic allocation
	// For simplicity, use a macro here 
        __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        // Suppose to be Bs[blockDim.x][blockDim.y] but need dynamic allocation
        // For simplicity, use a macro here
        __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + m * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < blockDim.x; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = m * blockDim.y * by + blockDim.x * bx;
    C[c + m * ty + tx] = Csub;
}

void matrix_multiply(int* a, int* b, int* c, int m, int n, int block_size) {

    dim3 dimBlock(block_size, block_size); //so that blockDim.x = blockDim.y = block_size
    //Divide the target matrix c into blocks, each thread handles one entry
    dim3 dimGrid(m/dimBlock.x, m/dimBlock.y); // 0 <= blockIdx.x <= m/block_size (dimGrid.x), 0 <= blockIdx.y <= m/block_size (dimGrid.y)
    
    int *d_a, *d_b, *d_c;
    //Allocate memory on GPU
    cudaMalloc(&d_a, m * n * sizeof(int));
    cudaMalloc(&d_b, n * m * sizeof(int));
    cudaMalloc(&d_c, m * m * sizeof(int));
    printf("Allocate memory on GPU done!\n");

    //Copy memory from CPU to GPU
    cudaMemcpy(d_a, a, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * m * sizeof(int), cudaMemcpyHostToDevice);
    printf("Copy memory from CPU to GPU done!\n");

    //matrix_multiply_cuda<<<dimGrid, dimBlock>>> (d_a, d_b, d_c, m, n);
    matrix_multiply_tiling_cuda<<<dimGrid, dimBlock>>> (d_a, d_b, d_c, m, n);
    printf("Matrix multiplication done!\n");

    //Copy memory from GPU to CPU
    cudaMemcpy(c, d_c, m * m * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Copy memory from GPU to CPU done!\n");

    //Free memory on GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("Free memory on GPU done!\n");
}

int main(int argc, const char * argv[]) {
    
    if (argc <= 1) {
        printf("No input file specified!\n");
        return 1;
    }
    //Read input matrix
    FILE* fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("The file can not be opened or does not exist!\n");
        return 1;
    }
    
    int m, n;
    fscanf(fp, "%d %d\n", &m, &n);
    int* a = (int*)malloc(m * n * sizeof(int));
    int* b = (int*)malloc(n * m * sizeof(int));
    int* c = (int*)malloc(m * m * sizeof(int));
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            //b[j][i] = a[i][j] for simplicity
            fscanf(fp, "%d", &a[i*n+j]);
            b[j*m+i] = a[i*n+j];
        }
    }
    fclose(fp);
    
    int block_size = BLOCK_SIZE;
    
    matrix_multiply(a, b, c, m, n, block_size);
    printf("Get back to CPU!\n");

    bool flag = true;
    if (m<= 1 << 8)
    for (int i=0; i<m; i++) {
        for (int j=0; j<m; j++) {
            int cell = 0;
            for (int k=0; k<n; k++)
		cell += a[i*n+k]*b[k*m+j];
            if (cell-c[i*m+j] != 0){
		printf("Wrong answer!\n");
		flag = false;
	        break;
	    }
	}
        if (!flag) break;
    }
    if ((m<= 1<<8)&&(flag)) printf("Correct answer!\n");
    
    free(a);
    free(b);
    free(c);
    printf("Free memory on CPU done!\n");
    return 0;
}
