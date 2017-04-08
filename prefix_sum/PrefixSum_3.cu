#include <iostream>

#define N 32
#define LOG 5
#define BLOCKSIZE 8

/* prefix sum */

using namespace std;

__global__ void add(int* in, int* out, int offset, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	out[gid] = in[gid];
	if(gid >= offset)
		out[gid] += in[gid-offset]; 
}

__global__ void schedule(int* in, int* out, int n){
	
	int offset=1;
	while(offset<N){
		if(offset & 1 == 0){ //odd
			add(in, out, offset, n);
		} else { //even
			add(out, in, offset, n);
		}
		offset = (offset<<1);
	}
}

int main(){
	int *in, *out;
	int *d_in, *d_out;
	
	int size = N * sizeof(int);
	// Allocate space for host of in, out
	in = (int *)malloc(size);
	out = (int *)malloc(size);
	
	// Allocate space for device copies of in, out
	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);
	
	//initialize input;
	for(int i=0; i<N; i++){
		in[i] = 1;
		out[i] = 0;
	}
	
	// Copy inputs to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	
	// Launch kernel on GPU
	schedule<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE >>>(d_in, d_out, N);
	
	//Copy memory from GPU to CPU 
	cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
			
	for(int i=0; i<N; i++){
		cout<< i << " : " << in[i] << " , " << out[i] <<endl;
	}
	
	//Clean Host
	free(in); free(out);
	
	//Clean Device
	cudaFree(d_in);
	cudaFree(d_out);
	
	return 0;
}