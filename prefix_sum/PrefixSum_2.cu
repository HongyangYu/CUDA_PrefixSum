#include <iostream>

#define N 32
#define BLOCKSIZE 8

/* 一维数Radius相加 share memory*/

using namespace std;

__global__ void add(int* in, int* out, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	extern __shared__ int temp[];
	
	int pout = 0, pin = 1;
	temp[threadIdx.x + pout * n] = (threadIdx.x>0) ? in[threadIdx.x-1] : 0;
	__syncthreads();
	
	for(int offset=1; offset<n; offset=(offset<<1)){
		int t = pout;
		pout = pin;
		pin = t;
		
		if(threadIdx.x >= offset){
			temp[threadIdx.x + pout*n] += temp[threadIdx.x + pin*n - offset];
		} else {
			temp[threadIdx.x+pout*n] = temp[threadIdx.x+pin*n];
		}
		__syncthreads();
	}
	out[threadIdx.x] = temp[threadIdx.x+pout*n];
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
	add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE, (BLOCKSIZE<<1) >>>(d_in, d_out, N);
		
	//Copy memory from GPU to CPU 
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