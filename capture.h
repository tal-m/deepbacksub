#ifndef _CAPTURE_H_
#define _CAPTURE_H_

#include <time.h>
#include <pthread.h>
#include <opencv2/videoio.hpp>

// threaded capture shared state
typedef struct {
	cv::VideoCapture *cap;
	cv::Mat *grab;
	cv::Mat *raw;
	int64 cnt;
	pthread_mutex_t lock;
	pthread_t tid;
	struct timespec last;
	int w, h, rate;
	bool (*callback)(cv::Mat *raw, void *ctx);
	void *cb_ctx;
} capinfo_t;

capinfo_t *capture_init(const char* device, int *w, int *h, int debug);
cv::Mat *capture_frame(capinfo_t *pcap);
void capture_setcb(capinfo_t *pcap, bool (*cb)(cv::Mat *, void *), void *ctx);
void capture_stop(capinfo_t *pcap);

#endif // _CAPTURE_H_
