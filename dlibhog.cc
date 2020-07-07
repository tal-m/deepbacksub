// Find face(s) in input image, generate positive mask (1=>face)

#include <stdio.h>
#include <signal.h>
#include <time.h>

//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "dlibhog.h"

struct _hoginfo_t {
    dlib::frontal_face_detector det;
    cv::Mat prev;
    int debug;
};

hoginfo_t *hog_init(int debug) {
    hoginfo_t *phg = new hoginfo_t;
    phg->debug = debug;
    phg->det = dlib::get_frontal_face_detector();
    return phg;
}

bool hog_faces(hoginfo_t *phg, cv::Mat& img, cv::Mat& out) {
    // convert to dlib bgr image type
    dlib::cv_image<dlib::bgr_pixel> bgr(img);
    // detect faces!
    std::vector<dlib::rectangle> faces = phg->det(bgr);
    if (faces.size()>0) {
        // map faces to output mask
        out = cv::Mat::zeros(img.size(),CV_32FC1);
        for (size_t f=0; f<faces.size(); f++) {
            // weight centre of facial ellipse, corrects HOG offsets
            cv::Point cen (
                (5*faces[f].left()+6*faces[f].right())/11,
                (2*faces[f].top()+faces[f].bottom())/3
            );
            // stretch out axes to encompass whole face
            cv::Size axes (
                (faces[f].right()-faces[f].left())*0.55,
                (faces[f].bottom()-faces[f].top())*0.7
            );
            cv::ellipse( out, cen, axes, 0, 0, 360, cv::Scalar(1.0), cv::FILLED);
        }
        out.copyTo(phg->prev);
    } else if (!phg->prev.empty()) {
        phg->prev.copyTo(out);
    }
    return true;
}

void hog_stop(hoginfo_t *phg) {
    delete phg;
}
