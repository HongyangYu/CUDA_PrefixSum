#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <ctime>

#define SIZE 1000
#define THREAD 1000

using namespace std;

__global__
void multiply_kernel(float * m, float * n, float * ans) {
	for (int i = 0;i < SIZE;i++) {
		*(ans + (blockIdx.x*SIZE) + threadIdx.x) += *(m + (blockIdx.x*SIZE) + i) * *(n + (i*SIZE) + threadIdx.x);
	}
}

int main() {
	
	float *m,*n, *ans;

	cudaMallocManaged(&m, SIZE*SIZE * sizeof(float));
	cudaMallocManaged(&n, SIZE*SIZE * sizeof(float));
	cudaMallocManaged(&ans, SIZE*SIZE * sizeof(float));
	
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0;j < SIZE;j++) {
			*(m + i*SIZE + j) = 2.0f;//(float)rand()/(float)100;
			*(n + i*SIZE + j) = 2.0f;
		}
	}

	multiply_kernel <<<SIZE, SIZE>>> (m, n, ans);

	cudaDeviceSynchronize();

	for (int i = 0; i < SIZE; i++) {
		for (int j = 0;j < SIZE;j++) {
			//cout << *(ans + i*SIZE + j) << "\t";
			if (*(ans + i*SIZE + j) != 4000.00f) {
				cout << "Error" << endl;
				goto out;
			}
		}
	}
	out:
	return 0;
}