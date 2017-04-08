#include <iostream>

#define N 32
#define BLOCKSIZE 4

/* 一维数Radius相加 share memory*/

using namespace std;

__global__ void add(int* in, int* out, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	extern __shared__ int temp[];
	
	temp[threadIdx.x] = in[gid];
	
	for(int offset=1; offset<n; offset=(offset<<1)){
		__syncthreads();
		if(threadIdx.x >= offset){
			temp[threadIdx.x] += temp[threadIdx.x-offset];
		} else if(gid >= offset){
			temp[threadIdx.x] += in[gid-offset];
		}
		__syncthreads(); //can only control threads in a block.
		in[gid] = temp[threadIdx.x];
	}
	out[gid] = in[gid];
	//out = in;
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

	// Launch add() kernel on GPU	
	add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE, BLOCKSIZE >>>(d_in, d_out, N);
		
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