BASEDIR :=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
INC-DIR := "-I$(BASEDIR)"

CC := g++
CFLAGS := -c -std=c++11 -O3 -funroll-all-loops  -Wno-deprecated
CUDA-FLAGS   = -ptx -std=c++11


MACHINE := rtx


ifeq ($(MACHINE),rtx)
  # anka-rtx
  TARGET := cuda-flow2d-rtx
  CUDA-TOP     = /home/ws/fe0968/local/cuda-10.0
  CUDA         = $(CUDA-TOP)/bin/nvcc 
  CUDA-INC-DIR = -I$(CUDA-TOP)/include 
  CUDA-LIB-DIR = -L$(CUDA-TOP)/lib64 -lcudart -lcuda  
else
  # ankaimage-concert
  TARGET := cuda-flow2d-concert
  CUDA         = /usr/bin/nvcc 
  CUDA-INC-DIR = -I/usr/include
  CUDA-LIB-DIR = -L/usr/lib/x86_64-linux-gnu -lcudart -lcuda 
endif






# $(wildcard *.cpp /xxx/xxx/*.cpp): get all .cpp files from the current directory and dir "/xxx/xxx/"
SRCS := $(wildcard $(BASEDIR)/src/*.cpp \
			$(BASEDIR)/src/*/*/*.cpp \
			$(BASEDIR)/src/*/*.cpp)
# $(patsubst %.cpp,%.o,$(SRCS)): substitute all ".cpp" file name strings to ".o" file name strings
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

CUDAOBJECTS 	= $(CUDASOURCES:.cu=.ptx)

#OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)
 
 $(TARGET): $(OBJS) cuda
	$(CC) $(OBJS) $(CUDA-LIB-DIR) -o $@

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
