#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define MATRIX_WIDTH 1024
#define PRINT_LIMIT 64
#define TOLERANCE 0.0001
#include "tools.h"
#include "strassen.h"


int main(int argc, char **argv){
    printf("\n\t********** GEMM Algorithm **********\n\n");
    double t1, t2;
    if(argc != 6){
        fprintf(stderr, "run as ./prog dev n nstop nt alg\nalg:\n\
    Classic             0\n\
    Blocked             1\n\
    CBLAS               2\n\
    Strassen CUBLAS     3\n\n");
        exit(1);
    }
    // 1) GET ARGS
    int dev                 = atoi(argv[1]);
    long n                  = atoi(argv[2]);
    int nstop               = atoi(argv[3]);
    int nt                  = atoi(argv[4]);
    int alg                 = atoi(argv[5]);
    long nelem              = n*n;
    double TFLOP = 2.0*n*n*n*1E-12;
    omp_set_num_threads(nt);




    // 2) CREATE AND FILL A B C matrices
    float *A = (float *)malloc(nelem * sizeof(float));
    float *B = (float *)malloc(nelem * sizeof(float));
    float *C = (float *)malloc(nelem * sizeof(float));
    float *goldC = (float *)malloc(nelem * sizeof(float));
    printf("Fill matrices A, B, C       "); fflush(stdout);
    t1 = omp_get_wtime();
    fillElements(A, n, 20.f);
    fillElements(B, n, 0.f);
    constMatrix(C, n, 0);
    constMatrix(goldC, n, 0);
    t2 = omp_get_wtime();
    printf("done: %f secs\n", t2-t1); fflush(stdout);
    #ifdef DEBUG
    printMatrix(A, n, "mat A");
    printMatrix(B, n, "mat B");
    #endif
    



    // 3) MATMUL (STRASSEN AND CLASSIC)
    //omp_set_dynamic(0);
    //omp_set_nested(4);
    //omp_set_num_threads(4);
    printf("%-28s", algorithms[alg]); fflush(stdout);
    t1 = omp_get_wtime();
    switch(alg){
        case 0:
            MatmulBasic(A, B, C, n);
            break;
        case 1:
            MatmulBlock(A, B, C, n);
            break;
        case 2:
            MatmulCBLAS(A, B, C, n);
            break;
        case 3:
            MatmulStrassenGPU(A, B, C, n, nstop);
            break;
    }
    t2 = omp_get_wtime();
    printf("done: %f secs (%f TFLOPS)\n", t2-t1, TFLOP/(t2-t1)); fflush(stdout);





    // 4) VERIFY RESULTS
    printf("[GOLD] %-21s", algorithms[2]); fflush(stdout);
    t1 = omp_get_wtime();
    MatmulCBLAS(A, B, goldC, n);
    t2 = omp_get_wtime();
    printf("done: %f secs (%f TFLOPS)\n", t2-t1, TFLOP/(t2-t1)); fflush(stdout);
#ifdef DEBUG
    printMatrix(C, n, "mat C");
    printMatrix(goldC, n, "mat goldC");
#endif
    printf("Verify (TOL = %f)     ", TOLERANCE); fflush(stdout);
    printf("%s\n", VerifyResults(C, goldC, n) ? "pass" : "fail"); fflush(stdout);
    double accError = 0.0f;
    #ifdef DEBUG
        for(int i = 0; i < nelem; ++i){
            accError += fabs(C[i] - goldC[i]);
        }
        printf("Accum Error: %f\n", accError);
    #endif



    // 5) CLEANUP
    free(A);
    free(B);
    free(C);
    free(goldC);
}
