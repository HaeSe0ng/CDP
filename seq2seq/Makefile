TARGET=main
TARGET_NOCDP=main_nocdp
TARGET_NOCDP_RDC=main_nocdp_rdc
OBJECTS=seq2seq_link.o seq2seq.o tvm_kernels_judy01.o util.o main.o
OBJECTS_NOCDP=seq2seq_nocdp.o tvm_kernels_judy01_nocdp.o util.o main.o
OBJECTS_NOCDP_RDC=seq2seq_nocdp_link.o seq2seq_nocdp.o tvm_kernels_judy01.o util.o main.o


CC=g++
CCFLAGS=-std=c++14 -Wall -O3
LDLIBS=-L/usr/local/cuda/lib64 -lcudart
NVCCFLAGS=-std=c++14 -arch=sm_61 

seq2seq.o: seq2seq.cu
	nvcc $(NVCCFLAGS) -O3 -rdc=true -c $^

tvm_kernels_judy01.o: tvm_kernels_judy01.cu
	nvcc $(NVCCFLAGS) -O3 -rdc=true -c $^

tvm_kernels_judy01_nocdp.o: tvm_kernels_judy01.cu
	nvcc $(NVCCFLAGS) -O3 -c -o $@ $^

seq2seq_link.o: seq2seq.o tvm_kernels_judy01.o
	nvcc $(NVCCFLAGS) -dlink -o $@ $^

seq2seq_nocdp.o: seq2seq_nocdp.cu reduce.cuh
	nvcc $(NVCCFLAGS) -O3 -c -o $@ seq2seq_nocdp.cu

seq2seq_nocdp_link.o: seq2seq_nocdp.o tvm_kernels_judy01.o
	nvcc $(NVCCFLAGS) -dlink -o $@ $^

main.o : main.cpp

all: $(TARGET) $(TARGET_NOCDP)

$(TARGET): $(OBJECTS)
	$(CC) $(CCFLAGS) $(OBJECTS) $(LDLIBS) -lcudadevrt -o $(TARGET)

$(TARGET_NOCDP): $(OBJECTS_NOCDP)
	$(CC) $(CCFLAGS) $(OBJECTS_NOCDP) $(LDLIBS) -o $(TARGET_NOCDP)

$(TARGET_NOCDP_RDC): $(OBJECTS_NOCDP_RDC)
	$(CC) $(CCFLAGS) $(OBJECTS_NOCDP_RDC) $(LDLIBS) -o $(TARGET_NOCDP_RDC)

clean:
	rm -rf $(TARGET) $(OBJECTS) $(TARGET_NOCDP) $(OBJECTS_NOCDP) $(TARGET_NOCDP_RDC) $(OBJECTS_NOCDP_RDC)

run: $(TARGET)
	./$(TARGET) $(ARGS)

run_nocdp: $(TARGET_NOCDP)
	./$(TARGET_NOCDP) $(ARGS)

run_nocdp_rdc: $(TARGET_NOCDP_RDC)
	./$(TARGET_NOCDP_RDC) $(ARGS)