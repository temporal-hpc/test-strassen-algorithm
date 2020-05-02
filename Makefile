DEBUG=no
PARAMS= -fopenmp
all:
	g++ -D${DEBUG} ${PARAMS} -O3 main.cpp -o prog
