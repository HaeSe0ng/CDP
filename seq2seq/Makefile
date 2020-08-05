TARGET=main
TARGET_NOCDP=main_nocdp
TARGET_NOCDP_RDC=main_nocdp_rdc
OBJECTS=seq2seq_link.o seq2seq.o tvm_kernels_elsa12.o torch_kernels.o util.o main.o
OBJECTS_NOCDP=seq2seq_nocdp.o tvm_kernels_elsa12_nocdp.o torch_kernels_nocdp.o util.o main.o
OBJECTS_NOCDP_RDC=seq2seq_nocdp_link.o seq2seq_nocdp.o tvm_kernels_elsa12.o torch_kernels.o util.o main.o


CC=g++
CCFLAGS=-std=c++14 -Wall -O3
LDLIBS=-L/usr/local/cuda/lib64 -lcudart
NVCCFLAGS=-std=c++14 -arch=sm_61 -O3

seq2seq.o: seq2seq.cu
	nvcc $(NVCCFLAGS) -rdc=true -c $^

tvm_kernels_elsa12.o: tvm_kernels_elsa12.cu
	nvcc $(NVCCFLAGS) -rdc=true -c $^

tvm_kernels_elsa12_nocdp.o: tvm_kernels_elsa12.cu
	nvcc $(NVCCFLAGS) -c -o $@ $^

torch_kernels.o: torch_kernels.cu
	nvcc $(NVCCFLAGS) -rdc=true -c $^

torch_kernels_nocdp.o: torch_kernels.cu
	nvcc $(NVCCFLAGS) -c -o $@ $^

seq2seq_link.o: seq2seq.o tvm_kernels_elsa12.o torch_kernels.o
	nvcc $(NVCCFLAGS) -dlink -o $@ $^

seq2seq_nocdp.o: seq2seq_nocdp.cu
	nvcc $(NVCCFLAGS) -c -o $@ $^

seq2seq_nocdp_link.o: seq2seq_nocdp.o tvm_kernels_elsa12.o torch_kernels.o 
	nvcc $(NVCCFLAGS) -dlink -o $@ $^

main.o : main.cpp

all: $(TARGET) $(TARGET_NOCDP) $(TARGET_NOCDP_RDC)

$(TARGET): $(OBJECTS)
	$(CC) $(CCFLAGS) $(OBJECTS) $(LDLIBS) -lcudadevrt -o $(TARGET)

$(TARGET_NOCDP): $(OBJECTS_NOCDP)
	$(CC) $(CCFLAGS) $(OBJECTS_NOCDP) $(LDLIBS) -o $(TARGET_NOCDP)

$(TARGET_NOCDP_RDC): $(OBJECTS_NOCDP_RDC)
	$(CC) $(CCFLAGS) $(OBJECTS_NOCDP_RDC) $(LDLIBS) -lcudadevrt -o $(TARGET_NOCDP_RDC)

clean:
	rm -rf $(TARGET) $(OBJECTS) $(TARGET_NOCDP) $(OBJECTS_NOCDP) $(TARGET_NOCDP_RDC) $(OBJECTS_NOCDP_RDC)

run: $(TARGET)
	./$(TARGET) $(ARGS)

run_nocdp: $(TARGET_NOCDP)
	./$(TARGET_NOCDP) $(ARGS)

run_nocdp_rdc: $(TARGET_NOCDP_RDC)
	./$(TARGET_NOCDP_RDC) $(ARGS)