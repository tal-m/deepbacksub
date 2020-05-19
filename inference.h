#ifndef _INFERENCE_H_
#define _INFERENCE_H_


// opaque type for callers
struct _tfinfo_t;
typedef struct _tfinfo_t tfinfo_t;

// tensor buffer info
typedef struct {
	int w, h, c;
	float *data;
} tfbuffer_t;
#define TFINFO_BUF_IN	0
#define TFINFO_BUF_OUT	1

tfinfo_t *tf_init(const char *modelname, int threads, int debug);
tfbuffer_t *tf_get_buffer(tfinfo_t *ptf, int which);
bool tf_infer(tfinfo_t *ptf);
void tf_stop(tfinfo_t *ptf);

#endif // _INFERENCE_H_
