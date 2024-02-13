#include "SIFT.h"
int main()
{
    cv::Mat img = cv::imread("./images/1.jpg");
    cv::imshow("1",img);
    cv::waitKey(0);

    cv::Mat img_ = cv::imread("./images/1_.jpg");
    cv::imshow("1",img_);
    cv::waitKey(0);

    SIFT::Keypoints keypoints,keypoints_;
    SIFT::extract(img,keypoints);
    SIFT::extract(img_,keypoints_);

    SIFT::Keypoints m1,m2;
    SIFT::match(keypoints,keypoints_,m1,m2);
    cv::Mat res = SIFT::linkPoints(img,m1,img_,m2);
    cv::imshow("1",res);
    cv::waitKey(0);

    return 0;
}
