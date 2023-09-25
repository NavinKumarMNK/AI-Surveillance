CC := g++
NVCC := /usr/local/cuda/bin/nvcc

CFLAGS := -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS += -I/opt/nvidia/deepstream/deepstream-6.3/sources/includes/ -I/usr/local/cuda/include

CUFLAGS := -std=c++14 -shared 
CUFLAGS += -I/opt/nvidia/deepstream/deepstream-6.3/sources/includes/ -I/usr/local/cuda/include

LIBS := -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS := -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS := $(wildcard *.h)
SRCFILES := nvdsyolofaceinferparser_cpp.cpp\
			nvdsyolofaceinferparser_cuda.cu

TARGET_LIB := ../libs/libnvdsinfer_yoloface.so

TARGET_OBJS := $(SRCFILES:.cpp=.o)
TARGET_OBJS := $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $(CUFLAGS)  $<

$(TARGET_LIB): $(TARGET_OBJS)
	$(CC) $(LFLAGS) $(TARGET_OBJS)  -o $@
	make clean

clean:
	rm -rf *.o
