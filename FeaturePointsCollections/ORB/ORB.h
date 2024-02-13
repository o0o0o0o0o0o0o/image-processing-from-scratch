//
// author: 会飞的吴克
//


#ifndef IMAGE_PROCESSING_FROM_SCRATCH_ORB_H
#define IMAGE_PROCESSING_FROM_SCRATCH_ORB_H
#include <opencv2/opencv.hpp>
namespace ORB {

    struct Keypoint{

        Keypoint()= default;
        Keypoint(const double& _x, const double& _y, const double& _size, const double& _theta,
                 const double& _response, int _o, int _sign):x(_x),y(_y),size(_size),theta(_theta),
                                                             response(_response),o(_o),sign(_sign){}

        double x,y;
        double size;
        double theta;
        double response;
        int o;
        int sign;
        unsigned char descriptor[32];
    };
    typedef std::vector<Keypoint> Keypoints;
    void extract(const cv::Mat& img, Keypoints& keypoints);
    void match(const Keypoints& src1, const Keypoints& src2, Keypoints& match1, Keypoints& match2, int max_pairs = 10);
    cv::Mat linkPoints(const cv::Mat& img1,const Keypoints& match1,
                       const cv::Mat& img2,const Keypoints& match2);

}


#endif //IMAGE_PROCESSING_FROM_SCRATCH_ORB_H
