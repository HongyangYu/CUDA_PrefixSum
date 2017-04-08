#include <iostream>

#define N 32
#define LOG 5
#define BLOCKSIZE 8

/* 一维数Radius相加 share memory*/

using namespace std;

__global__ void add(int* in, int d, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	int pre = (d==0) ? 1 : (2<<(d-1));
	
	if(gid >= pre) {
		in[gid] += in[gid-pre]; 
	}
}

int main(){
	int *in, *out;
	int *d_a;
	
	int size = N * sizeof(int);
	// Allocate space for host of in, out
	in = (int *)malloc(size);
	out = (int *)malloc(size);
	
	// Allocate space for device copies of in
	cudaMalloc((void **)&d_a, size);
	
	
	//initialize input;
	for(int i=0; i<N; i++){
		in[i] = 1;
		out[i] = 0;
	}
	
	
	// Launch add() kernel on GPU
	for(int d=0; d<=LOG; d++){
		// Copy inputs to device
		cudaMemcpy(d_a, in, size, cudaMemcpyHostToDevice);
		add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_a, d, N);
		cudaMemcpy(in, d_a, size, cudaMemcpyDeviceToHost);
	}
	
	//Copy memory from GPU to CPU 
	cudaMemcpy(out, d_a, size, cudaMemcpyDeviceToHost);
			
	for(int i=0; i<N; i++){
		cout<< i << " : " << in[i] << " , " << out[i] <<endl;
	}
	
	//Clean Host
	free(in); free(out);
	
	//Clean Device
	cudaFree(d_a);
	
	return 0;
}