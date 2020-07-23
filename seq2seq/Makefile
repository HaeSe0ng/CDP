TARGET=main
TARGET_NOCDP=main_nocdp
OBJECTS=seq2seq_link.o seq2seq.o util.o main.o
OBJECTS_NOCDP=seq2seq_nocdp.o util.o main.o

CC=g++
CCFLAGS=-std=c++11 -Wall -O3
LDLIBS=-L/usr/local/cuda/lib64 -lcudart
NVCCFLAGS=-std=c++11 -arch=sm_61 

seq2seq.o: seq2seq.cu
	nvcc $(NVCCFLAGS) -O3 -rdc=true -c $^ -lcudadevrt 

seq2seq_link.o: seq2seq.o
	nvcc $(NVCCFLAGS) -dlink -o $@ $^ -lcudadevrt

seq2seq_nocdp.o: seq2seq_nocdp.cu
	nvcc $(NVCCFLAGS) -O3 -c -o $@ $^

main.o : main.cpp

all: $(TARGET) $(TARGET_NOCDP)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDLIBS) -lcudadevrt -o $(TARGET)

$(TARGET_NOCDP): $(OBJECTS_NOCDP)
	$(CC) $(OBJECTS_NOCDP) $(LDLIBS) -o $(TARGET_NOCDP)

clean:
	rm -rf $(TARGET) $(OBJECTS) $(TARGET_NOCDP) $(OBJECTS_NOCDP)

run: $(TARGET)
	./$(TARGET) $(ARGS)

run_nocdp: $(TARGET_NOCDP)
	./$(TARGET_NOCDP) $(ARGS)