#pragma once
int getMove(){
    int ret = 0, sup=MATRIX_WIDTH;
    while(sup/2>=2){
        sup= sup/2;
        ret++;
    }
    return ret;
}

// print matrix in 2D visual form
void printMatrix(int *M, int n, const char *msg){
    if(n > 64){ return; }
    printf("%s:\n", msg);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            long index = i*n + j;
            printf("%i ", M[index]);
        }
        printf("\n");
    }
    printf("\n");
}

// fill matrix with sequence of 0,1,2,3,4 numbers
void fillElements(int *M, int n){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            long index = i*n + j;
            M[index] = (i + j + 1)%5;   
        }
    }
}

void constMatrix(int *M, int n, const int val){
    for(long i = 0; i < (long)(n*n); ++i){
        M[i] = val;   
    }
}

// C = A + B
void AddMatrix(int *A, int *B, int *C, int size){
    for(int i = 0; i < size; ++i){
        C[i] = A[i] + B[i];
    }
}

// C = A - B
void SubMatrix(int *A, int *B, int *C, int size){
    for(int i = 0; i < size; ++i){
        C[i] = A[i] - B[i];
    }
}

// C = A x B
void MatmulMatrix(int *A, int *B, int *C, int width){
    for (int i = 0; i < width; ++i){   
        for (int j = 0; j < width; ++j){
            int acc=0;
            for (int k = 0; k < width; ++k){
                acc += A[width*i + k] * B[width*k + j];
            }
            C[width*i + j] = acc;
        }
    }
}

bool VerifyResults(int *C1, int *C2, int n){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            long index = i*n + j;
            if(C1[index] != C2[index]){
                #ifdef DEBUG
                    fprintf(stderr, "Verify: error at A[i=%i, j=%i] = %i  !=   B[i=%i, j=%i] = %i\n", i, j, C1[index], i, j, C2[index]);
                #endif
                return false;
            }
        }
    }   
    return true;
}


