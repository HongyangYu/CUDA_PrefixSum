#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void sum(double* a, double* b, const int n) {
    //Given an array a[0...n-1], you need to compute b[0...n-1],
    //where b[i] = (i+1)*a[0] + i*a[1] + ... + 2*a[i-1] + a[i]
    //Note that b is NOT initialized with 0, be careful!
    //Write your CUDA code starting from here
    //Add any functions (e.g., device function) you want within this file
    
    return;
}

int read_write_vector(int argc, const char* argv[]) {

    //Read input from input file specified by user
    FILE* fp = fopen(argv[1], "r");
    if (fp == NULL) {
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
    if (fp == NULL) {
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

int read_write_matrix(int argc, const char* argv[]) {

    FILE *fp = fopen(argv[1], "w");
    if (fp == NULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    int n = 1 << 10; //column number
    int m = 1 << 10; //row number
    fprintf(fp, "%d %d\n", m, n);
    srand(time(NULL));
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++)
            fprintf(fp, "%d ", (rand() % 100));
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Finished writing\n");
    return 0;
}

int main(int argc, const char * argv[]) {

    if (argc != 2) {
        printf("The argument is wrong! Execute your program with only input file name!\n");
        return 1;
    }
    
    return read_write_matrix(argc, argv);
    //return read_write_vector(argc, argv);
}
