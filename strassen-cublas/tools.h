#define bsize 8

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
    if(n > 64){ return; }
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

void TrasposeMatrix(float *M, float *bt, int width){
    
    for(int i = 0; i < width; ++i){
        for(int j = 0; j < width; ++j){
            bt[width*j + i] = M[width* i + j];
        }
    }
}

void MatmulMatrix(float *A, float *B, float *C, int width){
    //#pragma omp for
    //printf("HERE BLOCK MATMUL");
    int n = width;
    int num = width/bsize;
    float *bt = (float *)malloc(n*n*sizeof(float));
    //printMatrix(B, width, "B");
    TrasposeMatrix(B, bt, width);
    //printMatrix(bt, width, "bt");
    //printf("HERE READY FOR BLOCK\n");
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
    free(bt);
}

bool VerifyResults(float *C1, float *C2, int n){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            long index = i*n + j;
            if(fabs(C1[index]-C2[index])>0.01){
                #ifdef DEBUG
                    fprintf(stderr, "Verify: error at A[i=%i, j=%i] = %f  !=   B[i=%i, j=%i] = %f\n", i, j, C1[index], i, j, C2[index]);
                #endif
                return false;
            }
        }
    }   
    return true;
}