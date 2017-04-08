#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCKSIZE 1024

//initialized the out array
__global__ void init(double* out, int n){
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	out[gid] = 0.0;
}

__global__ void add(double* in, double* out, int offset, int n){
	
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if(gid >= n) return ;
	
	out[gid] = in[gid];
	if(gid >= offset)
		out[gid] += in[gid-offset]; 
}

void launch(double* in, double* out, const int n){

	int size = n * sizeof(double);
	
	// Allocate space for device copies of in, out
	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);
	
	// Copy inputs to device
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	
	cout<<"Parallel Calcualtion Start"<<endl;
	clock_t t1 = clock();
	
	
	// Launch init() kernel on GPU
	init<<<(n+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_out, n);
	
	// Launch add() kernel on GPU
	int isOdd = 1;
	//prefix sum once
	for(int offset=1; offset<n; offset = (offset<<1)){
		if(isOdd == 1){ //odd
			add<<<(n+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_in, d_out, offset, n);
			isOdd = 0;
		} else { //even
			add<<<(n+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_out, d_in, offset, n);
			isOdd = 1;
		}
	}
	//prefix sum twice
	for(int offset=1; offset<n; offset = (offset<<1)){
		if(isOdd == 1){ //odd
			add<<<(n+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_in, d_out, offset, n);
			isOdd = 0;
		} else { //even
			add<<<(n+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_out, d_in, offset, n);
			isOdd = 1;
		}
	}
	
	clock_t t2 = clock();
	cout<<"Parallel Calcualtion End"<<endl;
	
	double time_spent = (double)(t2 - t1) / CLOCKS_PER_SEC * 1000;
	cout << "The running time of parallel addtition is " << time_spent << " miliseconds.\n" << endl;
	
	//Copy memory from GPU to CPU 
	if(isOdd==0){
		cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(out, d_in, size, cudaMemcpyDeviceToHost);
	}
		
	//Clean Device
	cudaFree(d_in);
	cudaFree(d_out);
}


void sum(double* a, double* b, const int n) {
    //Given an array a[0...n-1], you need to compute b[0...n-1],
    //where b[i] = (i+1)*a[0] + i*a[1] + ... + 2*a[i-1] + a[i]
    //note that b is nOT initialized with 0, be careful!
    //Write your CUDA code starting from here
    //Add any functions (e.g., device function) you want within this file
	
	launch(a, b, n);
	
    return;
}

int main(int argc, const char * argv[]) {

    if (argc != 2) {
        printf("The argument is wrong! Execute your program with only input file name!\n");
        return 1;
    }
    
    //Dummy code for creating a random input vectors
    //Convenient for the text purpose
    //Please comment out when you submit your code!!!!!!!!! 	
    /*FILE *fp = fopen(argv[1], "w");
    if (fp == nULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    int n = 1 << 24;
    fprintf(fp, "%d\n", n);
    srand(time(nULL));
    for (int i=0; i<n; i++)
        fprintf(fp, "%lg\n", ((double)(rand() % n))/100);
    fclose(fp);
    printf("Finished writing\n");*/
    
    //Read input from input file specified by user
    FILE* fp = fopen(argv[1], "r");
    if (fp == nULL) {
        printf("The file can not be opened or does not exist!\n");
        return 1;
    }
    int n;
    fscanf(fp, "%d\n", &n);
    printf("%d\n", n);
    double* a = malloc(n*sizeof(double));
    double* b = malloc(n*sizeof(double));
    for (int i=0; i<n; i++) {
        fscanf(fp, "%lg\n", &a[i]);
    }
    fclose(fp);
    
    //Main function
    sum(a, b, n);
    
    //Write b into output file
    fp = fopen("output.txt","w");
    if (fp == nULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    fprintf(fp, "%d\n", n);
    for (int i=0; i<n; i++)
        fprintf(fp, "%lg\n", b[i]);
    fclose(fp);
    free(a);
    free(b);
    printf("Done...\n");
    return 0;
}
