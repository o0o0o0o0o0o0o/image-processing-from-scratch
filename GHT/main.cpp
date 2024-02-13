//
// author: 会飞的吴克
//

#include "Generalized_Hough_Transform_sparse.h"

void test_GHT()
{
    cv::Mat img = cv::imread("./GHT/2.png");
    cv::Mat pattern = cv::imread("./GHT/2_.png");

//    auto I = GHT_SPARSE::generalhoughtransform(img,pattern,{255,0,0},3,{30,30,10,10},
//                                                0.3,180,{0.3,1.5},24,
//                                                {-GHT_SPARSE::PI/2,0},180); // 1

    auto I = GHT_SPARSE::generalhoughtransform(img,pattern,{255,0,0},1,{30,30,10,10},
                                                0.9,180,{2,5},60,
                                                {0,GHT_SPARSE::PI/2},180); // 2

//    auto I = GHT_SPARSE::generalhoughtransform(img,pattern,{0,0,0},1,{30,30,10,10},
//                                                0.4,180,{1.2,4},56,
//                                                {-GHT_SPARSE::PI,GHT_SPARSE::PI},180); // 3

//    auto I = GHT_SPARSE::generalhoughtransform(img,pattern,{255,215,0},1,{5,5,1,2},
//                                                0.5,180,{0.8,1.2},4,
//                                                {0,0},1); // 4


    cv::imshow("result",I);
    cv::waitKey(0);
    cv::imwrite("res.png",I);
}


int main() {
    test_GHT();
    return 0;
}