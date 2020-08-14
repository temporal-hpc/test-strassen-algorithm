#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define MATRIX_WIDTH 1024
#define PRINT_LIMIT 64
#include "tools.h"
#include "strassen.h"


int main(int argc, char **argv){
    printf("\n\t********** Strassen Algorithm **********\n\n");
    double timer;
    if(argc != 5){
        fprintf(stderr, "run as ./prog n nt print MIN_SIDE_MATRIX\n");
        exit(1);
    }
    int width = atoi(argv[1]);
    int nt = atoi(argv[2]);
    int print = atoi(argv[3]);
    int minside = atoi(argv[4]);
    int total = width * width;
    float *matrix_A = (float *)malloc(total * sizeof(float));
    float *matrix_B = (float *)malloc(total * sizeof(float));
    float *matrix_CStr = (float *)malloc(total * sizeof(float));
    float *matrix_CNor = (float *)malloc(total * sizeof(float));

    printf("Fill matrices A, B and C.............."); fflush(stdout);
    fillElements(matrix_A, width, 20.f);
    fillElements(matrix_B, width, 0.f);
    constMatrix(matrix_CStr, width, 0);
    constMatrix(matrix_CNor, width, 0);
    printf("done\n"); fflush(stdout);
    #ifdef DEBUG
    if(print){
        printMatrix(matrix_A, width, "mat A");
        printMatrix(matrix_B, width, "mat B");
    }
    
    //printMatrix(matrix_CStr, width, "Mat C (Strassen)");
    //printMatrix(matrix_CNor, width, "Mat C (Classic)");
    #endif
    
    //omp_set_dynamic(0);
    //omp_set_nested(4);
    //omp_set_num_threads(4);

    printf("Strassen Matmul......................."); fflush(stdout);
    timer = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    {
        #pragma omp single
        {
            StrassenAlgorithm(matrix_A, matrix_B, matrix_CStr, total, width, 0, minside);
        }
    }
    
    timer = omp_get_wtime() - timer;
    printf("done: %f secs\n", timer); fflush(stdout);
    
    // (2) classic Matmul
    printf("Classic Matmul........................"); fflush(stdout);
    timer = omp_get_wtime();
    #pragma omp parallel for num_threads(nt) 
        for (int i = 0; i < width; ++i){   
            for (int j = 0; j < width; ++j){
                float acc=0;
                for (int k = 0; k < width; ++k){
                    acc += matrix_A[width*i + k] * matrix_B[width*k + j];
                }
                matrix_CNor[width*i + j] = acc;
            }
        }
    timer = omp_get_wtime() - timer;
    printf("done: %f secs\n", timer); fflush(stdout);

    // (3) verify Results
#ifdef DEBUG
    printf("\n");
    if(print){
        printMatrix(matrix_CStr, width, "Strassen Matmul");
        printMatrix(matrix_CNor, width, "Classic Matmul");
    }
    
#endif
    printf("Verifying Strassen....................%s\n", VerifyResults(matrix_CStr, matrix_CNor, width) == true? "Pass" : "Fail");
    float acc = 0.f;
    int i, n = width;
    for(i = 0; i < total; ++i){
        if(matrix_CStr[i]==matrix_CNor[i]) acc++;
    }
    printf("Accuracy: %f\n", (acc/(total))*100);
    // (4) cleanup
    free(matrix_A);
    free(matrix_B);
    free(matrix_CStr);
    free(matrix_CNor);
}