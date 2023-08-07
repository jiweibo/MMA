
HOST_COMPILER ?= g++
NVCC := nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS := -m64
CCFLAGS := -std=c++11
LDFLAGS :=

INCLUDES := -I../common

SMS ?= 75
GENCODE_FLAGS := -gencode arch=compute_75,code=compute_75



all: build

build: reduction

reduction.o: reduction.cc
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_kernel.o: reduction_kernel.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction: reduction.o reduction_kernel.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+

clean:
	rm -f reduction reduction.o reduction_kernel.o