//
// author: 会飞的吴克
//

#include "WaterShed.h"

void test_WaterShed()
{
//    cv::Mat img = cv::imread("./WaterShed/1.jpeg");
//    auto res = Segmentation::WaterShed(img);
//    cv::imwrite("./WaterShed/result.jpg",res);
//    cv::imshow("res",res);
//    cv::waitKey(0);

//    Segmentation::WaterShedVideo("./WaterShed/BAD APPLE BIN.avi");

    cv::Mat img = cv::imread("./WaterShed/1.jpeg");
    auto res = Segmentation::WaterShedWithMarkers(img);
    cv::imwrite("./WaterShed/result.jpg",res);
    cv::imshow("res",res);
    cv::waitKey(0);
}

int main() {
    test_WaterShed();
    return 0;
}