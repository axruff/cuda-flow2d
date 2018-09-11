BASEDIR :=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
INC-DIR := "-I$(BASEDIR)"

CC := g++
CFLAGS := -c -std=c++11 -O3 -funroll-all-loops  -Wno-deprecated
TARGET := cuda-flow2d

CUDA-TOP     = /usr/local/cuda-7.0
CUDA         = $(CUDA-TOP)/bin/nvcc 
#CUDA         = $/usr/bin/nvcc                          # ESRF lbs191 GPU machine

CUDA-INC-DIR = -I$(CUDA-TOP)/include                    
CUDA-LIB-DIR = -L$(CUDA-TOP)/lib64 -lcudart -lcuda      

#CUDA-INC-DIR = -I/usr/include                               # ESRF lbs191 GPU machine
#CUDA-LIB-DIR = -L/usr/lib/x86_64-linux-gnu/ -lcudart -lcuda   # ESRF lbs191 GPU machine

CUDA-FLAGS   = -ptx -std=c++11




SRCS := $(wildcard $(BASEDIR)/src/*.cpp \
			$(BASEDIR)/src/*/*/*.cpp \
			$(BASEDIR)/src/*/*.cpp)

OBJS := $(patsubst %.cpp,%.o,$(SRCS))

CUDAOBJECTS 	= $(CUDASOURCES:.cu=.ptx)

#OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS) cuda
	$(CC) $(INC-DIR) $(CUDA-LIB-DIR) $(OBJS) -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR)  $< -o $@

cuda: 
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/add_2d.cu -o $(BASEDIR)/kernels/add_2d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/median_2d.cu -o $(BASEDIR)/kernels/median_2d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/convolution_2d.cu -o $(BASEDIR)/kernels/convolution_2d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/registration_2d.cu -o $(BASEDIR)/kernels/registration_2d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/resample_2d.cu -o $(BASEDIR)/kernels/resample_2d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/solve_2d.cu -o $(BASEDIR)/kernels/solve_2d.ptx
	$(CUDA) $(CUDA-FLAGS) $(INC-DIR) $(CUDA-INC-DIR) $(CUDA-LIB-DIR) $(BASEDIR)/src/kernels/correlation_2d.cu -o $(BASEDIR)/kernels/correlation_2d.ptx


clean:
	rm -rf $(TARGET) $(OBJS) $(CUDAOBJECTS)
	
.PHONY: all clean
