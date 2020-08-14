#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MATRIX_WIDTH 1024
#define PRINT_LIMIT 64
#include "tools.h"
#include "strassen.h"


int main(int argc, char **argv){
    printf("\n\t********** Strassen Algorithm **********\n\n");
    double timer;
    if(argc != 2){
        fprintf(stderr, "run as ./prog n\n");
        exit(1);
    }
    int width = atoi(argv[1]);
    int total = width * width;
    int *matrix_A = (int *)malloc(total * sizeof(int));
    int *matrix_B = (int *)malloc(total * sizeof(int));
    int *matrix_CStr = (int *)malloc(total * sizeof(int));
    int *matrix_CNor = (int *)malloc(total * sizeof(int));
    printf("Fill matrices A, B and C.............."); fflush(stdout);
    fillElements(matrix_A, width);
    fillElements(matrix_B, width);
    constMatrix(matrix_CStr, width, 0);
    constMatrix(matrix_CNor, width, 0);
    printf("done\n"); fflush(stdout);
    #ifdef DEBUG
        //printMatrix(matrix_A, width, "mat A");
        //printMatrix(matrix_B, width, "mat B");
        //printMatrix(matrix_CStr, width, "Mat C (Strassen)");
        //printMatrix(matrix_CNor, width, "Mat C (Classic)");
    #endif
    
    printf("Strassen Matmul......................."); fflush(stdout);
    timer = omp_get_wtime();
    StrassenAlgorithm(matrix_A, matrix_B, matrix_CStr, total, width);
    timer = omp_get_wtime() - timer;
    printf("done: %f secs\n", timer); fflush(stdout);


    // (2) classic Matmul
    printf("Classic Matmul........................"); fflush(stdout);
    timer = omp_get_wtime();
    MatmulMatrix(matrix_A,matrix_B, matrix_CNor, width);
    timer = omp_get_wtime() - timer;
    printf("done: %f secs\n", timer); fflush(stdout);

    // (3) verify Results
#ifdef DEBUG
    printf("\n");
    printMatrix(matrix_CStr, width, "Strassen Matmul");
    printMatrix(matrix_CNor, width, "Classic Matmul");
#endif
    printf("Verifying Strassen....................%s\n", VerifyResults(matrix_CStr, matrix_CNor, width) == true? "Pass" : "Fail");

    // (4) cleanup
    free(matrix_A);
    free(matrix_B);
    free(matrix_CStr);
    free(matrix_CNor);
}
