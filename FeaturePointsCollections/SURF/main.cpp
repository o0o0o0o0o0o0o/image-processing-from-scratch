//
// author: 会飞的吴克
//
#include "SURF.h"

int main()
{
    cv::Mat img = cv::imread("./images/2.jpg");
    cv::imshow("1",img);
    cv::waitKey(0);

    cv::Mat img_ = cv::imread("./images/2_.jpg");
    cv::imshow("1",img_);
    cv::waitKey(0);

    SURF::Keypoints keypoints, keypoints_;
    SURF::extract(img,keypoints);
    SURF::extract(img_,keypoints_);
    std::cout << keypoints.size() << std::endl;
    std::cout << keypoints_.size() << std::endl;

    SURF::Keypoints m1,m2;
    SURF::match(keypoints,keypoints_,m1,m2);
    cv::Mat res = SURF::linkPoints(img,m1,img_,m2);
    cv::imshow("1",res);
    cv::waitKey(0);

    return 0;
}
