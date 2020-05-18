#ifndef _INFERENCE_H_
#define _INFERENCE_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace tflite;

typedef struct {
	std::unique_ptr<tflite::FlatBufferModel> model;
	std::unique_ptr<Interpreter> interpreter;
	struct {
		int w, h, c;
		float *data;
	} buffers[2];
} tfinfo_t;
#define TFINFO_BUF_IN	0
#define TFINFO_BUF_OUT	1

tfinfo_t *tf_init(const char *modelname, int threads, int debug);
bool tf_infer(tfinfo_t *ptf);
void tf_stop(tfinfo_t *ptf);

#endif // _INFERENCE_H_
