//
// author: 会飞的吴克
//

#include "ORB.h"
int main(){

    cv::Mat img = cv::imread("./images/1.jpg");
    cv::imshow("1",img);
    cv::waitKey(0);

    cv::Mat img_ = cv::imread("./images/1_.jpg");
    cv::imshow("1",img_);
    cv::waitKey(0);

    ORB::Keypoints keypoints,keypoints_;
    ORB::extract(img,keypoints);
    ORB::extract(img_,keypoints_);

    ORB::Keypoints m1,m2;
    ORB::match(keypoints,keypoints_,m1,m2);
    cv::Mat res = ORB::linkPoints(img,m1,img_,m2);
    cv::imshow("1",res);
    cv::waitKey(0);

    return 0;
}
