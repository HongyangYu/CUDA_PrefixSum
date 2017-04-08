#include <iostream>

#define N 10000003 //10,000,000  1024 is ok
#define BLOCKSIZE 1024

/* prefix sum */

using namespace std;

void add_seq(double* in_seq, double* out_seq) {
	for(int i=0; i<N; i++){
		out_seq[i] += in_seq[i];
	}
	for(int i=0; i<N; i++){
		out_seq[i] += out_seq[i];
	}
}

__global__ void add(double* in, double* out, int offset, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	out[gid] = in[gid];
	if(gid >= offset)
		out[gid] += in[gid-offset]; 
}

int main(){
	double *in, *out, *in_seq, *out_seq;
	double *d_in, *d_out;
	
	int size = N * sizeof(double);
	// Allocate space for host of in, out
	in = (double *)malloc(size);
	out = (double *)malloc(size);
	in = (double *)malloc(size);
	out = (double *)malloc(size);
	
	// Allocate space for device copies of in, out
	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);
	
	//initialize input;
	for(int i=0; i<N; i++){
		in[i] = 1.0;
		out[i] = 0.0;
		in_seq[i] = 1.0;
		out_seq[i] = 0.0;
	}
	
	// Copy inputs to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	
	cout<<"Parallel Calcualtion Start"<<endl;
	
	clock_t t1 = clock();
	
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
	
	clock_t t2 = clock();
	
	cout<<"Parallel Calcualtion End"<<endl;
	
	double time_spent = (double)(t2 - t1) / CLOCKS_PER_SEC * 1000;
	cout << "The running time of parallel addtition is " << time_spent << " miliseconds." << endl;
	
	//Copy memory from GPU to CPU 
	cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++){
		if(isOdd==0){
			if(out[i]!=(i+1L)*(i+2L)/2L) 
				cout<< i << " : " << out[i] <<endl;
		} else {
			if(in[i]!=(i+1L)*(i+2L)/2L) 
				cout<< i << " : " << in[i] <<endl;
		}
	}
	
	add_seq(in_seq, out_seq);
	for(int i=0; i<N; i++){
		if(out_seq[i]!=(i+1L)*(i+2L)/2L)
			cout<<i<<","<<out_seq[i]<<endl;
	}
	
	cout<< "+=============+"<<endl;
	cout<< "N = " << N <<" , N*(N+1)/2 = "<< N*(N+1L)/2L <<endl;
	
	if(isOdd==0){
		cout<< (N-1) << " : " << (long)out[N-1] <<endl;
	} else {
		cout<< (N-1) << " - " << (long)in[N-1] <<endl;
	}
	
	//Clean Device
	cudaFree(d_in);
	cudaFree(d_out);

	//Clean Host
	free(in); free(out);
	
	
	return 0;
}