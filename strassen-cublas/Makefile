DEBUG=no
PARAMS= -fopenmp
COMP= -Xcompiler
all:
	nvcc -lcublas -lcblas -D${DEBUG} ${COMP} ${PARAMS} -O3 main.cu -o prog
