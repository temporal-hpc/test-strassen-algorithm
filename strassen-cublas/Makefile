DEBUG=DEBUG
PARAMS= -fopenmp
COMP= -Xcompiler
all:
	nvcc -lcublas -D${DEBUG} ${COMP} ${PARAMS} -O3 main.cu -o prog
