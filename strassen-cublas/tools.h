#define bsize 8

const char *algorithms[5] = {"Classic", "Blocked", "SGEMM CBLAS", "SGEMM CUBLAS", "Strassen+CUBLAS"};

void StrassenAlgorithm(float *A, float *B, float *C, long nelem, long n, int depth, int nstop, cublasHandle_t &handle);

void errChkCUBLAS(cublasStatus_t s, const char *msg){
    if(s != CUBLAS_STATUS_SUCCESS){
        printf("[CUBLAS] %s failed\n", msg);
        exit(EXIT_FAILURE);
    }
}



int getMove(){
    int ret = 0, sup=MATRIX_WIDTH;
    while(sup/2>=2){
        sup= sup/2;
        ret++;
    }
    return ret;
}

// print matrix in 2D visual form
void printMatrix(float *M, int n, const char *msg){
    if(n > PRINT_LIMIT){ return; }
    printf("%s:\n", msg);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            long index = i*n + j;
            printf("%.1f ", M[index]);
        }
        printf("\n");
    }
    printf("\n");
}

// fill matrix with sequence of 0,1,2,3,4 numbers
void fillElements(float *M, int n, float arg){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            long index = i*n + j;
            M[index] =  (float)rand()/RAND_MAX;
            //M[index] = (i + j + 1)%5 +arg;   
        }
    }
}

void constMatrix(float *M, int n, const int val){
    for(long i = 0; i < (long)(n*n); ++i){
        M[i] = val;   
    }
}

// C = A + B
void AddMatrix(float *A, float *B, float *C, int size){
    //#pragma omp parallel
    //{
        for(int i = 0; i < size; ++i){
            C[i] = A[i] + B[i];
        }
    //}
}

// C = A - B
void SubMatrix(float *A, float *B, float *C, int size){
    //#pragma omp parallel
    //{
        for(int i = 0; i < size; ++i){
            C[i] = A[i] - B[i];
        }
    //}
}

// C = A x B
void TrasposeMatrix(float *M, float *bt, int n){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            bt[n*j + i] = M[n* i + j];
        }
    }
}

bool VerifyResults(float *C, float *goldC, int n){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            long index = i*n + j;
            if(fabs(goldC[index] - C[index])/goldC[index] > TOLERANCE){
                #ifdef DEBUG
                    fprintf(stderr, "Verify: error at C[i=%i, j=%i] = %f  !=   goldC[i=%i, j=%i] = %f\n", i, j, C[index], i, j, goldC[index]);
                #endif
                return false;
            }
        }
    }   
    return true;
}

double MatmulBasic(float *A, float *B, float *C, int n){
    double t1, t2;
    t1 = omp_get_wtime();
    #pragma omp parallel for 
    for (int i = 0; i < n; ++i){   
        for (int j = 0; j < n; ++j){
            float acc=0;
            unsigned long q = n*i + j;
            for (int k = 0; k < n; ++k){
                acc += A[n*i + k] * B[n*k + j];
            }
            C[q] = acc;
        }
    }
    t2 = omp_get_wtime();
    return t2-t1;
}

double MatmulBlock(float *A, float *B, float *C, int n){
    double t1, t2;
    //#pragma omp for
    //printf("HERE BLOCK MATMUL");
    int num = n/bsize;
    float *bt = (float *)malloc(n*n*sizeof(float));
    //printMatrix(B, n, "B");
    TrasposeMatrix(B, bt, n);
    //printMatrix(bt, n, "bt");
    //printf("HERE READY FOR BLOCK\n");
    t1 = omp_get_wtime();
    for (int i = 0; i < num; ++i){
            for (int j = 0; j < num; ++j){
                for (int k = 0; k < bsize; ++k){
                    for (int m = 0; m < bsize; ++m){
                        double sum = 0.0;
                        for (int r = 0; r < num; r++){
                            for (int p = 0; p < bsize; ++p){
                                //int i1 = i*bsize*n + r*bsize + k*n + p;
                                //int i2 = j*bsize*n + r*bsize*n + m + p*n;
                                //printf("I1: %i I2: %i\n",i1,i2);
                                //printf(" %i  %i  %i  %i  %i  %i  %i  %i     %i   %i \n",i,j,k,m,r,p,n,bsize,i1,i2);
                                //sum+=a[i*bsize*n + r*bsize + k*n + p]*b[j*bsize + r*bsize*n + m + p*n];
                                sum+=A[i*bsize*n + r*bsize + k*n + p]*bt[j*bsize*n + r*bsize + m*n + p];
                                //printf("HERE\n");
                            }
                        }
                        C[i*bsize*n + j*bsize + k*n + m] = sum;
                    }
                }
            }
        }
    //printf("HERE ENDED BLOCK\n");
    t2 = omp_get_wtime();
    free(bt);
    return t2-t1;
}

double MatmulCBLAS(float *A, float *B, float *C, long n){
    double t1, t2;
    t1 = omp_get_wtime();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0, C, n);
    t2 = omp_get_wtime();
    return t2-t1;
}

double MatmulStrassenGPU(float *A, float *B, float *C, long n, long nstop, cublasHandle_t &handle){
    double t1, t2;
    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            StrassenAlgorithm(A, B, C, n*n, n, 0, nstop, handle);
        }
    }
    t2 = omp_get_wtime();
    return t2-t1;
}

double MatmulCUBLAS(float *hA, float *hB, float *hC, long n, float *alpha, float *beta, cublasHandle_t &handle){
    double t1, t2;
    float *dA, *dB, *dC;
    long nelem = n*n;
    cublasStatus_t stat;
    cudaMalloc((void**)&dA, nelem*sizeof(*hA));
    cudaMalloc((void**)&dB, nelem*sizeof(*hB));
    cudaMalloc((void**)&dC, nelem*sizeof(*hC)); 
    stat = cublasSetMatrix(n, n, sizeof(*hA), hA, n, dA, n);
    errChkCUBLAS(stat, "MatmulCUBLAS cublasSetMatrix hA -> dA");
    stat = cublasSetMatrix(n, n, sizeof(*hB), hB, n, dB, n);
    errChkCUBLAS(stat, "MatmulCUBLAS cublasSetMatrix hB -> dB");
    stat = cublasSetMatrix(n, n, sizeof(*hC), hC, n, dC, n);
    errChkCUBLAS(stat, "MatmulCUBLAS cublasSetMatrix hC -> dC");
    t1 = omp_get_wtime();
    //stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, dB, n, dA, n, beta, dC, n);
    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha,
                          dB, CUDA_R_32F, n,
                          dA, CUDA_R_32F, n,
                          beta, dC, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    t2 = omp_get_wtime();
    errChkCUBLAS(stat, "MatmulCUBLAS GemmEx");
    //stat = cublasGetMatrix(n, n, sizeof(*hC), dC, n, hC, n);
    stat = cublasGetVector(nelem, sizeof(*hC), dC, 1, hC, 1);
    errChkCUBLAS(stat, "MatmulCUBLAS cublas copy dC -> hC");
    return t2-t1;
}
