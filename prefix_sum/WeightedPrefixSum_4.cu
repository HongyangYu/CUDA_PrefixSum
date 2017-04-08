#include <iostream>

#define N 10000003 //10,000,000  1024 is ok
#define BLOCKSIZE 1024

/* prefix sum */

using namespace std;

__global__ void add(double* in, double* out, int offset, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	out[gid] = in[gid];
	if(gid >= offset)
		out[gid] += in[gid-offset]; 
}

void add_seq(double* in_seq, double* out_seq) {
	for(int i=0; i<N; i++){
		out_seq[i] += in_seq[i];
	}
	for(int i=0; i<N; i++){
		out_seq[i] += out_seq[i];
	}
}

int main(){
	double *in, *out, *in_seq, *out_seq;
	double *d_in, *d_out;
	
	int size = N * sizeof(double);
	// Allocate space for host of in, out
	in = (double *)malloc(size);
	out = (double *)malloc(size);
	in_seq = (double *)malloc(size);
	out_seq = (double *)malloc(size);
	
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
	
	
	// Sequential Calcualtion
	cout<<"Sequential Calcualtion Start"<<endl;
	clock_t t3 = clock();
	
	add_seq(in_seq, out_seq);
	
	clock_t t4 = clock();
	double time_spent2 = (double)(t4 - t3) / CLOCKS_PER_SEC * 1000;
	cout << "The running time of sequential addtition is " << time_spent2 << " miliseconds." << endl;
	cout<<"Sequential Calcualtion End"<<endl;
	
	// check:
	cout<< "+=============+"<<endl;
	cout<< "N = " << N <<endl;
	cout<< "+=============+"<<endl;
	
	for(long i=0; i<N; i++){
		if(isOdd==1){
			if((long)out[i]!=(long)out_seq[i]) 
				cout<< i << " : " << out[i] <<endl;
		} else {
			if((long)in[i]!=(long)out_seq[i]) 
				cout<< i << " : " << in[i] <<endl;
		}
	}
	
	//Clean Device
	cudaFree(d_in);
	cudaFree(d_out);

	//Clean Host
	free(in); free(out);
	
	
	return 0;
}