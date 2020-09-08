
CFLAGS = -Ofast -march=native -fno-trapping-math -fassociative-math -funsafe-math-optimizations -Wall -pthread
LDFLAGS = -lrt -ldl

# TensorFlow
TFBASE=../tensorflow.git
TFLITE=$(TFBASE)/tensorflow/lite/tools/make/
CFLAGS += -I $(TFBASE) -I $(TFLITE)/downloads/absl -I $(TFLITE)/downloads/flatbuffers/include
LDFLAGS += -L $(TFLITE)/gen/linux_x86_64/lib/ -ltensorflow-lite

# DLib - lord knows why we have to specify full paths to libblas & liblapack..
# if we try -lblas -llapack, it b0rks with unknown symbols or libraries
CFLAGS += -std=c++11 -mavx
LDFLAGS += -ldlib -lX11 /usr/lib/x86_64-linux-gnu/libblas.so.3 /usr/lib/x86_64-linux-gnu/liblapack.so.3

# git clone -b v2.1.0  https://github.com/tensorflow/tensorflow $(TFBASE)
# cd $(TFBASE)/tensorflow/lite/tools/make
# ./download_dependencies.sh && ./build_lib.sh

# OpenCV
ifeq ($(shell pkg-config --exists opencv; echo $$?), 0)
    CFLAGS += $(shell pkg-config --cflags opencv)
    LDFLAGS += $(shell pkg-config --libs opencv) -ldl
else ifeq ($(shell pkg-config --exists opencv4; echo $$?), 0)
    CFLAGS += $(shell pkg-config --cflags opencv4)
    LDFLAGS += $(shell pkg-config --libs opencv4) -ldl
else
    $(error Couldn't find OpenCV)
endif

deepseg: deepseg.cc loopback.cc capture.cc inference.cc dlibhog.cc
	g++ $^ ${CFLAGS} ${LDFLAGS} -o $@

all: deepseg

clean:
	-rm deepseg
