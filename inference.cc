// Wrapper for tensorflow inference processing
//
#include "inference.h"

#define ASSERT_OR_NULL(x) { if (!(x)) return NULL; }

tfinfo_t *tf_init(const char *modelname, int threads, int debug) {
	// Allocate info block
	tfinfo_t *ptf = new tfinfo_t;

	// Load model
	ptf->model = tflite::FlatBufferModel::BuildFromFile(modelname);
	ASSERT_OR_NULL(ptf->model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	InterpreterBuilder builder(*(ptf->model), resolver);
	builder(&ptf->interpreter);
	ASSERT_OR_NULL(ptf->interpreter != nullptr);

	// Allocate tensor buffers.
	ASSERT_OR_NULL(ptf->interpreter->AllocateTensors() == kTfLiteOk);

	// extract input & output buffer info
	for (int b=0; b<2; b++) {
		int tnum = (0==b) ? ptf->interpreter->inputs()[0] : ptf->interpreter->outputs()[0];
		TfLiteType t_type = ptf->interpreter->tensor(tnum)->type;
		ASSERT_OR_NULL(t_type == kTfLiteFloat32);

		TfLiteIntArray* dims = ptf->interpreter->tensor(tnum)->dims;
		if (debug) for (int i = 0; i < dims->size; i++) printf("tensor #%d: %d\n",tnum,dims->data[i]);
		ASSERT_OR_NULL(dims->data[0] == 1);
	
		ptf->buffers[b].h = dims->data[1];
		ptf->buffers[b].w = dims->data[2];
		ptf->buffers[b].c = dims->data[3];
		ptf->buffers[b].data = ptf->interpreter->typed_tensor<float>(tnum);
		ASSERT_OR_NULL(ptf->buffers[b].data != nullptr);
	}

	// set interpreter params
	ptf->interpreter->SetNumThreads(threads);
	ptf->interpreter->SetAllowFp16PrecisionForFp32(true);

	return ptf;
}

bool tf_infer(tfinfo_t *ptf) {
	return (ptf->interpreter->Invoke() == kTfLiteOk);
}

void tf_stop(tfinfo_t *ptf) {
}
