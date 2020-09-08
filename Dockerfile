FROM ubuntu:20.04  AS builder

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -f -y git curl unzip build-essential libx11-dev libopencv-dev libdlib-dev 

WORKDIR /opt

RUN git clone https://github.com/tensorflow/tensorflow.git tensorflow.git

COPY . deepbacksub

WORKDIR /opt/tensorflow.git

RUN ./tensorflow/lite/tools/make/download_dependencies.sh
RUN ./tensorflow/lite/tools/make/build_lib.sh

WORKDIR /opt/deepbacksub

RUN make

FROM ubuntu:20.04

WORKDIR /opt/deepbacksub

COPY --from=builder /opt/deepbacksub/deepseg /opt/deepbacksub/*.tflite ./

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -f -y libx11-6 libopencv-highgui4.2 libopencv-imgcodecs4.2 libopencv-imgproc4.2 libopencv-core4.2 libopencv-videoio4.2  libdlib19 
