#pragma once
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
        //int *firstMatrix = (int *)malloc(size*sizeof(int));
        //int *secondMatrix = (int *)malloc(size*sizeof(int));
        int *firstM1 = (int *)malloc(size*sizeof(int));
        int *secondM1 = (int *)malloc(size*sizeof(int));
        int *firstM2 = (int *)malloc(size*sizeof(int));
        int *firstM3 = (int *)malloc(size*sizeof(int));
        int *firstM4 = (int *)malloc(size*sizeof(int));
        int *firstM5 = (int *)malloc(size*sizeof(int));
        int *firstM6 = (int *)malloc(size*sizeof(int));
        int *secondM6 = (int *)malloc(size*sizeof(int));
        int *firstM7 = (int *)malloc(size*sizeof(int));
        int *secondM7 = (int *)malloc(size*sizeof(int));
        int *firstC11 = (int *)malloc(size*sizeof(int));
        int *secondC11 = (int *)malloc(size*sizeof(int));
        int *firstC22 = (int *)malloc(size*sizeof(int));
        int *secondC22 = (int *)malloc(size*sizeof(int));
        
        //auxiliar count to fill the submatrix
        int count = 0;
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

        //Seven calls of Strassen
        //First call
        //m1 = (a11 + a22)(b11 + b22)
        AddMatrix(a11, a22, firstM1, size);
        AddMatrix(b11, b22, secondM1, size);
        StrassenAlgorithm(firstM1, secondM1, m1, size, width);

        //Second call
        //m2 = (a21 + a22)b11
        AddMatrix(a21, a22, firstM2, size);
        StrassenAlgorithm(firstM2, b11, m2, size, width);

        //Third call
        //m3 = a11(b12 - b22)
        SubMatrix(b12, b22, firstM3, size);
        StrassenAlgorithm(a11, firstM3, m3, size, width);

        //Fourth call
        //m4 = a22(b21 - b11)
        SubMatrix(b21, b11, firstM4, size);
        StrassenAlgorithm(a22, firstM4, m4, size, width);

        //Fifth call
        //m5 = (a11 + a12)b22
        AddMatrix(a11, a12, firstM5, size);
        StrassenAlgorithm(firstM5, b22, m5, size, width);

        //Sixth call
        //m6 = (a21 - a11)(b11 + b12)
        SubMatrix(a21, a11, firstM6, size);
        AddMatrix(b11, b12, secondM6, size);
        StrassenAlgorithm(firstM6, secondM6, m6, size, width);

        //Seventh call
        //m7 = (a12 - a22)(b21 + b22)
        SubMatrix(a12, a22, firstM7, size);
        AddMatrix(b21, b22, secondM7, size);
        StrassenAlgorithm(firstM7, secondM7, m7, size, width);

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

        free(c11);
        free(c12);
        free(c21);
        free(c22);

    }
    

}
