#pragma once
#define LIMIT 2
//#define MINSIDE 8
//#define MINMATRIX MINSIDE*MINSIDE


void StrassenAlgorithm(float *A, float *B, float *C, long nelem, long n, int depth, int nstop){
    //printf("\nHere SEQ nelem: %i\n", nelem);
    if(n<=nstop){   
        //printf("IT (SEQ) SHOULD BE HERE\n");
        //For the minor case, its only a normal matrix multiplication
        //MatmulMatrix(A, B, C, 16);
        int MINMATRIX = nstop * nstop;
        double start, end;
        cublasHandle_t handle;
        cublasStatus_t stat;
        float *da, *db, *dc;
        float alpha = 1.0, beta = 0.0;
        cudaMalloc((void**)&da, MINMATRIX*sizeof(*A));
        cudaMalloc((void**)&db, MINMATRIX*sizeof(*B));
        cudaMalloc((void**)&dc, MINMATRIX*sizeof(*C)); 
        start = omp_get_wtime();
        stat = cublasCreate(&handle);   
        stat = cublasSetMatrix(nstop,nstop,sizeof(*A),B,nstop, da, nstop);
        stat = cublasSetMatrix(nstop,nstop,sizeof(*B),A,nstop, db, nstop);
        stat = cublasSetMatrix(nstop,nstop,sizeof(*C),C,nstop, dc, nstop);
        end = omp_get_wtime();
        #ifdef DEBUG
            printf("Allocation took: %f seconds\n", end - start);
        #endif

        start = omp_get_wtime();
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nstop, nstop, nstop, &alpha, da, nstop, db, nstop, &beta, dc, nstop);
        end = omp_get_wtime();
        #ifdef DEBUG
            printf("Operation took: %f seconds\n", end - start);
        #endif

        start = omp_get_wtime();
        stat = cublasGetMatrix(nstop,nstop, sizeof(*C), dc, nstop, C, nstop);
        end = omp_get_wtime();
        #ifdef DEBUG
            printf("GetMatrix took: %f seconds\n", end - start);
        #endif

        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        cublasDestroy(handle);
    }
    else{
        //printf("INITIALIZING SPLIT AND CONQUER, CALLING STRASSEN IN REGION(%i,%i)\n", wi, wj);
        //setup relevant variables
        //size is the size of the submatrix
        //hn is the half "n" for the n/2 x n/2 submatrices
        //stride is for operational purposes
        int size = nelem/4;
        int hn = n/2;
        int stride = n;

        //printf("\nnelem: %i, hn: %i, stride: %i\n",size, hn, stride);
        //printf("here\n");
        //Create submatrix of A
        float *a11 = (float *)malloc(size*sizeof(float));
        float *a12 = (float *)malloc(size*sizeof(float));
        float *a21 = (float *)malloc(size*sizeof(float));
        float *a22 = (float *)malloc(size*sizeof(float));
        //Create submatrix of B
        float *b11 = (float *)malloc(size*sizeof(float));
        float *b12 = (float *)malloc(size*sizeof(float));
        float *b21 = (float *)malloc(size*sizeof(float));
        float *b22 = (float *)malloc(size*sizeof(float));
        //Create submatrix of C
        float *c11 = (float *)malloc(size*sizeof(float));
        float *c12 = (float *)malloc(size*sizeof(float)); 
        float *c21 = (float *)malloc(size*sizeof(float));
        float *c22 = (float *)malloc(size*sizeof(float));
        //Create M(or P) components of Strassen algorithm
        float *m1 = (float *)malloc(size*sizeof(float));
        float *m2 = (float *)malloc(size*sizeof(float));
        float *m3 = (float *)malloc(size*sizeof(float));
        float *m4 = (float *)malloc(size*sizeof(float));
        float *m5 = (float *)malloc(size*sizeof(float));
        float *m6 = (float *)malloc(size*sizeof(float));
        float *m7 = (float *)malloc(size*sizeof(float));
        //Matrix to store operations
        //float *firstMatrix = (float *)malloc(size*sizeof(float));
        //float *secondMatrix = (float *)malloc(size*sizeof(float));
        float *firstM1 = (float *)malloc(size*sizeof(float));
        float *secondM1 = (float *)malloc(size*sizeof(float));
        float *firstM2 = (float *)malloc(size*sizeof(float));
        float *firstM3 = (float *)malloc(size*sizeof(float));
        float *firstM4 = (float *)malloc(size*sizeof(float));
        float *firstM5 = (float *)malloc(size*sizeof(float));
        float *firstM6 = (float *)malloc(size*sizeof(float));
        float *secondM6 = (float *)malloc(size*sizeof(float));
        float *firstM7 = (float *)malloc(size*sizeof(float));
        float *secondM7 = (float *)malloc(size*sizeof(float));
        float *firstC11 = (float *)malloc(size*sizeof(float));
        float *secondC11 = (float *)malloc(size*sizeof(float));
        float *firstC22 = (float *)malloc(size*sizeof(float));
        float *secondC22 = (float *)malloc(size*sizeof(float));
        
        //auxiliar count to fill the submatrix
        int count = 0;
        //build submatrix
        //printf("Here after sub creation\n");
        for(int i = 0; i < hn; ++i){
            for(int j = 0; j < hn; ++j){
                a11[count] = A[i*stride+j];
                a12[count] = A[hn+i*stride+j];
                a21[count] = A[stride*hn+i*stride + j];
                a22[count] = A[hn*stride+hn+i*stride+j];
                b11[count] = B[i*stride+j];
                b12[count] = B[hn+i*stride+j];
                b21[count] = B[stride*hn+i*stride + j];
                b22[count] = B[hn*stride+hn+i*stride+j];
                c11[count] = 0.0;
                c12[count] = 0.0;
                c21[count] = 0.0;
                c22[count] = 0.0;
                count++;
            }
        }
        //Seven calls of Strassen
        //First call
        //m1 = (a11 + a22)(b11 + b22)
        AddMatrix(a11, a22, firstM1, size);
        AddMatrix(b11, b22, secondM1, size);
        StrassenAlgorithm(firstM1, secondM1, m1, size, hn, depth + 1, nstop);

        //Second call
        //m2 = (a21 + a22)b11
        AddMatrix(a21, a22, firstM2, size);
        StrassenAlgorithm(firstM2, b11, m2, size, hn, depth + 1, nstop);

        //Third call
        //m3 = a11(b12 - b22)
        SubMatrix(b12, b22, firstM3, size);
        StrassenAlgorithm(a11, firstM3, m3, size, hn, depth + 1, nstop);

        //Fourth call
        //m4 = a22(b21 - b11)
        SubMatrix(b21, b11, firstM4, size);
        StrassenAlgorithm(a22, firstM4, m4, size, hn, depth + 1, nstop);

        //Fifth call
        //m5 = (a11 + a12)b22
        AddMatrix(a11, a12, firstM5, size);
        StrassenAlgorithm(firstM5, b22, m5, size, hn, depth + 1, nstop);
        

        //Sixth call
        //m6 = (a21 - a11)(b11 + b12)
        SubMatrix(a21, a11, firstM6, size);
        AddMatrix(b11, b12, secondM6, size);
        StrassenAlgorithm(firstM6, secondM6, m6, size, hn, depth + 1, nstop);
        

        //Seventh call
        //m7 = (a12 - a22)(b21 + b22)
        SubMatrix(a12, a22, firstM7, size);    
        AddMatrix(b21, b22, secondM7, size);  
        StrassenAlgorithm(firstM7, secondM7, m7, size, hn, depth + 1, nstop);

        free(a11);
        free(a12);
        free(a21);
        free(a22);
        free(b11);
        free(b12);
        free(b21);
        free(b22);
        free(firstM1);
        free(secondM1);
        free(firstM2);
        free(firstM3);
        free(firstM4);
        free(firstM5);
        free(firstM6);
        free(secondM6);
        free(firstM7);
        free(secondM7);

        //Apply m's matrix to c submatrix
        //c11 = m1 + m4 - m5 + m7
        AddMatrix(m1, m4, firstC11, size);
        SubMatrix(firstC11, m5, secondC11, size);
        AddMatrix(secondC11, m7, c11, size);

        //c12 = m3 + m5
        AddMatrix(m3, m5, c12, size);

        //c21 = m2 + m4
        AddMatrix(m2 , m4, c21, size);

        //c22 = m1 - m2 + m3 + m6
        SubMatrix(m1, m2, firstC22, size);
        AddMatrix(firstC22, m3, secondC22, size);        
        AddMatrix(secondC22, m6, c22, size);
    
        free(m1);
        free(m2);
        free(m3);
        free(m4);
        free(m5);
        free(m6);
        free(m7);
        free(firstC11);
        free(secondC11);
        free(firstC22);
        free(secondC22);

        count = 0;
        for(int i = 0; i < hn; ++i){
            for(int j = 0; j < hn; ++j){
                C[i*stride + j] = c11[count];
                C[hn + i*stride + j] = c12[count];
                C[hn*stride + i*stride + j] = c21[count];
                C[hn*stride + hn + i*stride + j] = c22[count];
                count++;
            }
        }

        free(c11);
        free(c12);
        free(c21);
        free(c22);
    }
}
