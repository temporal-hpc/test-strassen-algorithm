#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MATRIX_WIDTH 1024
#define ll long long

int getMove(){
    int ret = 0, sup=MATRIX_WIDTH;
    while(sup/2>=2){
        sup= sup/2;
        ret++;
    }
    return ret;
}

void printMatrix(int *M, int elements, int width){
    for(int i = 0; i < elements; ++i){
        printf("%f ", M[i]);
        if((i+1)%width == 0 && i>0) printf("\n");
    }
}

void fillElements(int *M, int elements){
    for(int i = 0; i < elements; ++i){
        M[i] = (i + 1)%5;   
    }
}
void AddMatrix(int *A, int *B, int *C, int size){
    for(int i = 0; i < size; ++i){
        C[i] = A[i] + B[i];
    }
    
}

void SubstractionMatrix(int *A, int *B, int *C, int size){
    for(int i = 0; i < size; ++i){
        C[i] = A[i] - B[i];
    }
}

void MatmulMatrix(int *A, int *B, int *C, int width){
    for (int i = 0; i < width; ++i){   
        for (int j = 0; j < width; ++j){
            for (int k = 0; k < width; ++k){
                C[width*j + i] += A[width*j + k] * B[width*k + i];
            }
        }
    }
}

float VerifyResults(int *A, int *B, ll size){
    int total = 0;
    for(ll i = 0; i < size; ++i){
        if(A[i] == B[i]) total++;
    }   
    return total;
}

void StrassenAlgorithm(int *A, int *B, int *C, int elements, int old_width){
    //printf("ELEMENTS: %i, REGION(%i,%i), MOVE %i\n", elements, wi, wj, move_index);
    
    if(elements/2 == 2)
    {   
        //For the minor case, its only a normal matrix multiplication
        MatmulMatrix(A, B, C, elements/2);
    }
    else
    {
        //printf("INITIALIZING SPLIT AND CONQUER, CALLING STRASSEN IN REGION(%i,%i)\n", wi, wj);
        //setup relevant variables
        //size is the size of the submatrix
        //width is the width of the submatrix
        //stride is for operational purposes
        int size = elements/4;
        int width = old_width/2;
        int stride = old_width;

        //printf("\nElements: %i, width: %i, stride: %i\n",size, width, stride);

        //Create submatrix of A
        int *a11 = (int *)malloc(size*sizeof(int));
        int *a12 = (int *)malloc(size*sizeof(int));
        int *a21 = (int *)malloc(size*sizeof(int));
        int *a22 = (int *)malloc(size*sizeof(int));
        //Create submatrix of B
        int *b11 = (int *)malloc(size*sizeof(int));
        int *b12 = (int *)malloc(size*sizeof(int));
        int *b21 = (int *)malloc(size*sizeof(int));
        int *b22 = (int *)malloc(size*sizeof(int));
        //Create submatrix of C
        int *c11 = (int *)malloc(size*sizeof(int));
        int *c12 = (int *)malloc(size*sizeof(int)); 
        int *c21 = (int *)malloc(size*sizeof(int));
        int *c22 = (int *)malloc(size*sizeof(int));
        //Create M(or P) components of Strassen algorithm
        int *m1 = (int *)malloc(size*sizeof(int));
        int *m2 = (int *)malloc(size*sizeof(int));
        int *m3 = (int *)malloc(size*sizeof(int));
        int *m4 = (int *)malloc(size*sizeof(int));
        int *m5 = (int *)malloc(size*sizeof(int));
        int *m6 = (int *)malloc(size*sizeof(int));
        int *m7 = (int *)malloc(size*sizeof(int));
        //Matrix to store operations
        int *firstMatrix = (int *)malloc(size*sizeof(int));
        int *secondMatrix = (int *)malloc(size*sizeof(int));
        //auxiliar count to fill the submatrix
        int count = 0;
        //printMatrix(A,elements,old_width);
        //build submatrix
        for(int i = 0; i < width; ++i){
            for(int j = 0; j < width; ++j){
                a11[count] = A[i*stride+j];
                a12[count] = A[width+i*stride+j];
                a21[count] = A[stride*width+i*stride + j];
                a22[count] = A[width*stride+width+i*stride+j];
                b11[count] = B[i*stride+j];
                b12[count] = B[width+i*stride+j];
                b21[count] = B[stride*width+i*stride + j];
                b22[count] = B[width*stride+width+i*stride+j];
                c11[count] = 0.0;
                c12[count] = 0.0;
                c21[count] = 0.0;
                c22[count] = 0.0;
                count++;
            }
        }

        //printMatrix(a11,size,width);
        //printf("\n\n");
        //printMatrix(a12,size,width);
        //printf("\n\n");
        //printMatrix(a21,size,width);
        //printf("\n\n");
        //printMatrix(a22,size,width);
        //printf("\n\n");
        //Seven calls of Strassen
        //First call
        //m1 = (a11 + a22)(b11 + b22)
        AddMatrix(a11, a22, firstMatrix, size);
        AddMatrix(b11, b22, secondMatrix, size);
        StrassenAlgorithm(firstMatrix, secondMatrix, m1, size, width);

        //Second call
        //m2 = (a21 + a22)b11
        AddMatrix(a21, a22, firstMatrix, size);
        StrassenAlgorithm(firstMatrix, b11, m2, size, width);

        //Third call
        //m3 = a11(b12 - b22)
        SubstractionMatrix(b12, b22, firstMatrix, size);
        StrassenAlgorithm(a11, firstMatrix, m3, size, width);

        //Fourth call
        //m4 = a22(b21 - b11)
        SubstractionMatrix(b21, b11, firstMatrix, size);
        StrassenAlgorithm(a22, firstMatrix, m4, size, width);

        //Fifth call
        //m5 = (a11 + a12)b22
        AddMatrix(a11, a12, firstMatrix, size);
        StrassenAlgorithm(firstMatrix, b22, m5, size, width);

        //Sixth call
        //m6 = (a21 - a11)(b11 + b12)
        SubstractionMatrix(a21, a11, firstMatrix, size);
        AddMatrix(b11, b12, secondMatrix, size);
        StrassenAlgorithm(firstMatrix, secondMatrix, m6, size, width);

        //Seventh call
        //m7 = (a12 - a22)(b21 + b22)
        SubstractionMatrix(a12, a22, firstMatrix, size);
        AddMatrix(b21, b22, secondMatrix, size);
        StrassenAlgorithm(firstMatrix, secondMatrix, m7, size, width);

        //free(a11);
        //free(a12);
        //free(a21);
        //free(a22);
        //free(b11);
        //free(b12);
        //free(b21);
        //free(b22);

        //Apply m's matrix to c submatrix
        //c11 = m1 + m4 - m5 + m7
        AddMatrix(m1, m4, firstMatrix, size);
        SubstractionMatrix(firstMatrix, m5, secondMatrix, size);
        AddMatrix(secondMatrix, m7, c11, size);

        //c12 = m3 + m5
        AddMatrix(m3, m5, c12, size);

        //c21 = m2 + m4
        AddMatrix(m2 , m4, c21, size);

        //c22 = m1 - m2 + m3 + m6
        SubstractionMatrix(m1, m2, firstMatrix, size);
        AddMatrix(firstMatrix, m3, secondMatrix, size);        
        AddMatrix(secondMatrix, m6, c22, size);
        
        
        free(m1);
        free(m2);
        free(m3);
        free(m4);
        free(m5);
        free(m6);
        free(m7);
        free(firstMatrix);
        free(secondMatrix);


        //printf("\n");
        //printMatrix(c11, size, width);
        //printf("\n");
        //printMatrix(c12, size, width);
        //printf("\n");
        //printMatrix(c21, size, width);
        //printf("\n");
        //printMatrix(c22, size, width);
        //printf("\n");

        //c sub matrix build C again
        count = 0;
        for(int i = 0; i < width; ++i){
            for(int j = 0; j < width; ++j){
                C[i*stride + j] = c11[count];
                C[width + i*stride + j] = c12[count];
                C[width*stride + i*stride + j] = c21[count];
                C[width*stride + width + i*stride + j] = c22[count];
                count++;
            }
        }

        //free(c11);
        //free(c12);
        //free(c21);
        //free(c22);
        //free(a11);
        //free(a12);
        //free(a21);
        //free(a22);
        //free(b11);
        //free(b12);
        //free(b21);
        //free(b22);
        
    }
    

}

int main(int argc, char **argv)
{
    int width = atoi(argv[1]);
    int total = width * width;
    clock_t timer;
    int *matrix_A = (int *)malloc(total * sizeof(int));
    int *matrix_B = (int *)malloc(total * sizeof(int));
    int *matrix_CStr = (int *)malloc(total * sizeof(int));
    int *matrix_CNor = (int *)malloc(total * sizeof(int));
    printf("Fill matrix\n");
    fillElements(matrix_A, total);
    fillElements(matrix_B, total);
    int move = getMove();
    //printf("move %i\n\n", move);

    //printMatrix(matrix_A, total, MATRIX_WIDTH);
    //printf("\n\n");
    //printMatrix(matrix_B, total, MATRIX_WIDTH);
    //printf("\n\n");
    printf("Begin compute\n");
    timer = clock();
    StrassenAlgorithm(matrix_A, matrix_B, matrix_CStr, total, width);
    timer = clock() - timer;
    printf("\n\nStrassen Algorithm matmul execution time: %f\n", ((double)timer)/CLOCKS_PER_SEC);
    //printMatrix(matrix_CStr, total, width);
    timer = clock();
    MatmulMatrix(matrix_A,matrix_B, matrix_CNor, width);
    timer = clock() - timer;
    printf("\n\nNormal algorithm matmul execution time: %f\n", ((double)timer)/CLOCKS_PER_SEC);
    //printMatrix(matrix_CNor, total, width);
    printf("Accuracy Strassen/Normal: %f\n", VerifyResults(matrix_CStr, matrix_CNor, total)/total*100);
    free(matrix_A);
    free(matrix_B);
    free(matrix_CStr);
    free(matrix_CNor);
}
