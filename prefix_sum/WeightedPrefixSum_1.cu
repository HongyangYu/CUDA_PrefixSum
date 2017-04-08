#include <iostream>

#define N 10000003 //10,000,000,  1024 is ok
#define BLOCKSIZE 1024

/* prefix sum */

using namespace std;

__global__ void add(int* in, int* out, int offset, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	out[gid] = in[gid];
	if(gid >= offset)
		out[gid] += in[gid-offset]; 
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
	int isOdd = 1;
	//prefix sum once
	for(int offset=1; offset<N; offset = (offset<<1)){
		if(isOdd == 1){ //odd
			add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_in, d_out, offset, N);
			isOdd = 0;
		} else { //even
			add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_out, d_in, offset, N);
			isOdd = 1;
		}
	}
	//prefix sum twice
	for(int offset=1; offset<N; offset = (offset<<1)){
		if(isOdd == 1){ //odd
			add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_in, d_out, offset, N);
			isOdd = 0;
		} else { //even
			add<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_out, d_in, offset, N);
			isOdd = 1;
		}
	}
	
	//Copy memory from GPU to CPU 
	cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
			
	for(int i=0; i<N; i++){
		if(isOdd==0){
			if(out[i]!=(i+1)*(i+2)/2) 
				cout<< i << " : " << out[i] <<endl;
		} else {
			if(in[i]!=(i+1)*(i+2)/2) 
				cout<< i << " : " << in[i] <<endl;
		}
		
	}
	
	cout<< "+=============+"<<endl;
	cout<< "N = " << N <<endl;
	if(isOdd==0){
		cout<< (N-1) << " : " << out[N-1] <<endl;
	} else {
		cout<< (N-1) << " - " << in[N-1] <<endl;
	}

	//Clean Host
	free(in); free(out);
	
	//Clean Device
	cudaFree(d_in);
	cudaFree(d_out);
	
	return 0;
}