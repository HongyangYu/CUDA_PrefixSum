#include <iostream>

#define N 32
#define BLOCKSIZE 4

/* 一维数Radius相加 share memory*/

using namespace std;

__global__ void add(int* in, int offset, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	extern __shared__ int temp[];
	
	temp[threadIdx.x] = in[gid];
	
	__syncthreads(); //can only control threads in a block.
	if(threadIdx.x >= offset){
		in[threadIdx.x] += temp[threadIdx.x-offset];
	} else if(gid >= offset){
		in[threadIdx.x] += in[gid-offset];
	}
	in[gid] = temp[threadIdx.x];
}

int main(){
	int *in;
	int *d_in;
	
	int size = N * sizeof(int);
	// Allocate space for host of in
	in = (int *)malloc(size);
	
	// Allocate space for device copies of in
	cudaMalloc((void **)&d_in, size);
	
	//initialize input;
	for(int i=0; i<N; i++){
		in[i] = 1;
	}
	
	// Copy inputs to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

	for(int offset=1; offset<N; offset=(offset<<1)){
		// Launch add() kernel on GPU	
		add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE >>>(d_in, offset, N);
	}
	//Copy memory from GPU to CPU 
	cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
			
	for(int i=0; i<N; i++){
		cout<< i << " : " << in[i] <<endl;
	}
	
	//Clean Host
	free(in); 
	
	//Clean Device
	cudaFree(d_in);
	
	return 0;
}