// OpenCV video capture thread wrapper
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>	// for various macro values

#include "capture.h"

// threaded capture state
struct _capinfo_t {
	cv::VideoCapture *cap;
	cv::Mat *grab;
	int64 cnt;
	pthread_mutex_t lock;
	pthread_t tid;
	struct timespec last;
	int w, h, rate;
	bool (*callback)(cv::Mat *, void *);
	void *cb_ctx;
};

// capture thread function
static void *grab_thread(void *arg) {
	capinfo_t *ci = (capinfo_t *)arg;
	bool done = false;
	// while we have a grab frame.. grab frames
	while (!done) {
		bool ok = ci->cap->grab();
		pthread_mutex_lock(&ci->lock);
		ci->cnt++;
		if (ci->grab!=NULL) {
			if (ok)
				ok = ci->cap->retrieve(*(ci->grab));
			if (ok && ci->callback!=NULL)
				ok = ci->callback(ci->grab, ci->cb_ctx);
		} else {
			done = true;
		}
		pthread_mutex_unlock(&ci->lock);
		// if we had grab, retrieve or callback failure, try looping
		if (!ok) {
			ci->cap->set(CV_CAP_PROP_POS_FRAMES, 0);
			ci->cnt = 0;
		}
		// ensure we wait until next expected frame
		// (or files whizz by in milliseconds)
		long ns = 1000000000L/ci->rate;
		long nx = ci->last.tv_nsec+ns;
		ci->last.tv_nsec = nx%1000000000;
		ci->last.tv_sec = ci->last.tv_sec+(nx/1000000000);
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ci->last, NULL);
	}
	return NULL;
}

capinfo_t *capture_init(const char *device, int *w, int *h, int *r, int debug) {
	// allocate capture info and contents
	capinfo_t *pcap = new capinfo_t;
	pcap->cap = new cv::VideoCapture;
	pcap->grab = new cv::Mat;
	pcap->cnt = 0;
	pcap->lock = PTHREAD_MUTEX_INITIALIZER;
	pcap->callback = NULL;
	pcap->cb_ctx = NULL;
	// check for local device name and ensure using V4L2, set capture props,
	// otherwise assume URL and allow OpenCV to choose the right backend,
	// finally, always enable RGB (actually BGR24) conversion so we have sane input
	// https://github.com/opencv/opencv/blob/master/modules/videoio/src/cap_v4l.cpp#1525
	if (strncmp(device, "/dev/video", 10)==0) {
		pcap->cap->open(device, CV_CAP_V4L2);
		pcap->cap->set(CV_CAP_PROP_FRAME_WIDTH,  *w);
		pcap->cap->set(CV_CAP_PROP_FRAME_HEIGHT, *h);
		pcap->cap->set(CV_CAP_PROP_CONVERT_RGB, true);
	} else {
		pcap->cap->open(device);
		pcap->cap->set(CV_CAP_PROP_CONVERT_RGB, true);
	}
	// always read the actual dimensions & rate back
	pcap->w=*w=(int)pcap->cap->get(CV_CAP_PROP_FRAME_WIDTH);
	pcap->h=*h=(int)pcap->cap->get(CV_CAP_PROP_FRAME_HEIGHT);
	pcap->rate=*r=(int)pcap->cap->get(CV_CAP_PROP_FPS);
	if (pcap->rate<0)
		pcap->rate=*r=30;	// default V4L2 rate (says OpenCV manual)
	clock_gettime(CLOCK_MONOTONIC, &pcap->last);
	// kick off separate grabber thread to keep OpenCV/FFMpeg happy (or it lags badly)
	if (pthread_create(&pcap->tid, NULL, grab_thread, pcap)) {
		return NULL;
	}
	return pcap;
}

void capture_frame(capinfo_t *pcap, cv::Mat& out) {
	// done?
	if (!pcap->grab)
		return;
	// wait for valid frame
	while (pcap->grab->empty()) {
		struct timespec ts = { 0, 1000000 }; // 1ms
		clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, NULL);
	}
	// copy buffer out under lock
	pthread_mutex_lock(&pcap->lock);
	pcap->grab->copyTo(out);
	pthread_mutex_unlock(&pcap->lock);
	return;
}

int64 capture_count(capinfo_t *pcap) {
	return pcap->cnt;
}

void capture_setcb(capinfo_t *pcap, bool (*cb)(cv::Mat *, void *), void *ctx) {
	pthread_mutex_lock(&pcap->lock);
	pcap->callback = cb;
	pcap->cb_ctx = ctx;
	pthread_mutex_unlock(&pcap->lock);
}

void capture_stop(capinfo_t *pcap) {
	pthread_mutex_lock(&pcap->lock);
	pcap->grab = NULL;
	pthread_mutex_unlock(&pcap->lock);
	pthread_join(pcap->tid, NULL);
}
