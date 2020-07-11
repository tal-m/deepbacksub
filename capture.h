#ifndef _CAPTURE_H_
#define _CAPTURE_H_

#include <opencv2/core/mat.hpp>

// opaque type for callers
struct _capinfo_t;
typedef struct _capinfo_t capinfo_t;

capinfo_t *capture_init(const char* device, int *w, int *h, int *r, int debug);
void capture_frame(capinfo_t *pcap, cv::Mat& out);
int64 capture_count(capinfo_t *pcap);
void capture_setcb(capinfo_t *pcap, bool (*cb)(cv::Mat *, void *), void *ctx);
void capture_stop(capinfo_t *pcap);

#endif // _CAPTURE_H_
