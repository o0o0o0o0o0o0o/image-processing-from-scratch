//
// author: 会飞的吴克
//

#ifndef IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H
#define IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H
#include <opencv2/opencv.hpp>
namespace DarkChannel {
#define __Assert__(x,msg)\
do\
{\
    if(!(x)){throw std::runtime_error((msg));}\
}while(false)

    cv::Mat dehaze(const cv::Mat& img);
}


#endif //IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H
