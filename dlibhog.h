#ifndef _DLIBHOG_H_
#define _DLIBHOG_H_


// opaque type for callers
struct _hoginfo_t;
typedef struct _hoginfo_t hoginfo_t;

// faces
hoginfo_t *hog_init(int debug);
bool hog_faces(hoginfo_t *phg, cv::Mat& img, cv::Mat& out);
void hog_stop(hoginfo_t *phg);

#endif // _DLIBHOG_H_
