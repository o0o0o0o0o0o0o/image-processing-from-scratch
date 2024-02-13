//
// author: 会飞的吴克
//

#include <iostream>
#include "ransac.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


void test_Ransac_line()
{
    std::vector<double> model;
    std::vector<point2D> Data;
    point2D D[30] = {{1,1.2},{1.12,1.02},{0.55,0.57},{2.11,2.05},{3.1,2.97},{2.32,2.28},{4.23,4.44},{4.44,4.5},{3.2,3.2},{4.73,4.68},
                     {5,1.2},{4,1.02},{0.55,1.2},{3.5,2.05},{3.1,4.21},{1,1.9},{2.2,3.2},{4.3,1.},{5.0,3.2},{4.73,4},
                     {5,3.},{3,1.54},{0.5,1.2},{3.5,5},{3.1,4.21},{1,2.3},{2.02,3.2},{7.3,6.},{4.626,3.2},{4.73,1.2}};

    for(unsigned long i = 0;i < 30;i++)
    {
        Data.push_back(D[i]);
    }

    model = Ransac_line(Data, true);
}

void test_Ransac_sift()
{
    cv::Mat img1 = cv::imread("./Ransac/2.jpg");
    cv::Mat img2 = cv::imread("./Ransac/2_.jpg");

    cv::Mat image1, image2;
    cv::cvtColor(img1,image1,cv::COLOR_RGB2GRAY);
    cv::cvtColor(img2,image2,cv::COLOR_RGB2GRAY);    // 读取图像灰度化

    cv::Ptr<cv::SiftFeatureDetector > detector = cv::SiftFeatureDetector::create();
    cv::Ptr<cv::SiftDescriptorExtractor > descriptorExtractor = cv::SiftDescriptorExtractor::create(); // cv::xfeatures2d:: for older versions of Opencv
    std::vector<cv::KeyPoint> kpt1,kpt2;
    detector->detect(image1,kpt1);
    detector->detect(image2,kpt2);
    cv::Mat descriptor1, descriptor2;
    descriptorExtractor->compute(image1,kpt1,descriptor1);
    descriptorExtractor->compute(image2,kpt2,descriptor2);    // 计算关键点和描述符

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matchPairs;
    std::vector<cv::DMatch> matchPairs2;

    matcher.match(descriptor1,descriptor2,matchPairs);      // 特征点匹配
    matcher.match(descriptor2,descriptor1,matchPairs2);
    int temp;
    for(unsigned long i = 0;i < matchPairs2.size();i++)
    {
        temp = matchPairs2[i].queryIdx;
        matchPairs2[i].queryIdx = matchPairs2[i].trainIdx;
        matchPairs2[i].trainIdx = temp;
    }
    matchPairs.insert(matchPairs.end(),matchPairs2.begin(),matchPairs2.end());

    std::sort(matchPairs.begin(),matchPairs.end());
    std::vector<cv::Point2f> goodkpts1, goodkpts2;
    std::cout << "pairs: " << matchPairs.size() << std::endl;
    unsigned long num = matchPairs.size()*0.05 > 60 ? matchPairs.size()*0.05 : 60;
    for(unsigned long i = 0;i < num;i++)
    {
        goodkpts1.push_back(kpt1[matchPairs[i].queryIdx].pt);
        goodkpts2.push_back(kpt2[matchPairs[i].trainIdx].pt);
    }                                                       // 保留前面比较优秀的匹配对，并转换点类型


    // 计算变换矩阵
    // cv::Mat transform = cv::findHomography(goodkpts1,goodkpts2,cv::RANSAC);
    std::vector<sift_pair> Data;
    for(unsigned long i = 0;i < goodkpts1.size();i++)
    {
        Data.push_back({goodkpts1[i].x,goodkpts1[i].y,goodkpts2[i].x,goodkpts2[i].y});
    }
    std::vector<double> model = Ransac_sift(Data,1.0/10);
    model.push_back(1);
    cv::Mat transform(model);
    transform = transform.reshape(0,3);


    // 计算输出图像大小，和基准图像锚点
    cv::Mat lt = transform * (cv::Mat_<double >(3,1) << 0.f,0.f,1.f);
    cv::Mat lb = transform * (cv::Mat_<double >(3,1) << 0.f,img1.rows-1,1.f);
    cv::Mat rt = transform * (cv::Mat_<double >(3,1) << img1.cols-1,0.f,1.f);
    cv::Mat rb = transform * (cv::Mat_<double >(3,1) << img1.cols-1,img1.rows-1,1.f);

    double minx = 0,maxx = img2.cols,miny = 0,maxy = img2.rows;
    double lt_x = lt.at<double >(0)/lt.at<double >(2); double lt_y = lt.at<double >(1)/lt.at<double >(2);
    double rb_x = rb.at<double >(0)/rb.at<double >(2); double rb_y = rb.at<double >(1)/rb.at<double >(2);
    double lb_x = lb.at<double >(0)/lb.at<double >(2); double lb_y = lb.at<double >(1)/lb.at<double >(2);
    double rt_x = rt.at<double >(0)/rt.at<double >(2); double rt_y = rt.at<double >(1)/rt.at<double >(2);

    double mx1, Mx1, mx2, Mx2, my1, My1, my2, My2;
    if(lt_x < rb_x){mx1 = lt_x;Mx1 = rb_x;} else{mx1 = rb_x;Mx1 = lt_x;}
    if(lb_x < rt_x){mx2 = lb_x;Mx2 = rt_x;} else{mx2 = rt_x;Mx2 = lb_x;}
    if(lt_y < rb_y){my1 = lt_y;My1 = rb_y;} else{my1 = rb_y;My1 = lt_y;}
    if(lb_y < rt_y){my2 = lb_y;My2 = rt_y;} else{my2 = rt_y;My2 = lb_y;}
    minx = std::min(minx,std::min(mx1,mx2)); maxx = std::max(maxx,std::max(Mx1,Mx2));
    miny = std::min(miny,std::min(my1,my2)); maxy = std::max(maxy,std::max(My1,My2));

    double output_size_x = maxx - minx , output_size_y = maxy - miny;
    double anchor_x = -minx , anchor_y = -miny;


    // 图像拼接
    cv::Mat img1_trans;
    cv::Mat shifter = (cv::Mat_<double >(3,3) << 1.f,0.f,anchor_x,
                                                 0.f,1.f,anchor_y,
                                                 0.f,0.f,1.f);
    cv::warpPerspective(img1,img1_trans,shifter*transform,cv::Size(output_size_x,output_size_y));


    // 做图像融合(在重叠部分做加权处理，使得过渡更加自然)
    cv::Mat result(img1_trans.rows,img1_trans.cols,CV_8UC3);
    result.setTo(0);
    // 这里我自重叠像素点向质心连线做垂线，根据垂足将连线的划分比例来加权
    // 计算质心，均匀质地的三角形质心公式x = (x1+x2+x3)/3; y = (y1+y2+y3)/3
    // 多边形可拆分成三角形，用各三角形面积将各三角形质心坐标加权求和归一化即可
    double centroid2_x = (2*anchor_x + img2.cols) / 2.0f, centroid2_y = (2*anchor_y + img2.rows) / 2.0f; // img2未经变换，是矩形，质心即为几何中心
    double centroid1_x = 0, centroid1_y = 0, Area = 0;
    double v1_x = rt_x - lt_x, v1_y = rt_y - lt_y, v2_x = rb_x - lt_x, v2_y = rb_y - lt_y; // 边向量
    double area = abs(v1_x*v2_y - v2_x*v1_y) / 2;
    centroid1_x += area*(v1_x + v2_x); centroid1_y += area*(v1_y + v2_y);
    Area += area;

    v1_x = lb_x - lt_x, v1_y = lb_y - lt_y;
    area = abs(v1_x*v2_y - v2_x*v1_y) / 2;
    centroid1_x += area*(v1_x + v2_x); centroid1_y += area*(v1_y + v2_y);
    Area += area;

    centroid1_x = centroid1_x / Area / 3 + lt_x + anchor_x;
    centroid1_y = centroid1_y / Area / 3 + lt_y + anchor_y;

    // 质心连线方程
    double A = centroid1_y - centroid2_y, B = -(centroid1_x - centroid2_x), C = centroid1_x*centroid2_y - centroid2_x*centroid1_y;
    double denomin = A*A+B*B;
    for(int i = 0;i < result.rows;i++)
    {
        for(int j = 0;j < result.cols;j++)
        {
            cv::Vec3b pixel1 = img1_trans.at<cv::Vec3b>(i,j);
            int im2_ind_row = i - anchor_y, im2_ind_col = j - anchor_x;

            if(0<=im2_ind_row&&im2_ind_row<img2.rows&&0<=im2_ind_col&&im2_ind_col<img2.cols)
            {
                cv::Vec3b pixel2 = img2.at<cv::Vec3b>(im2_ind_row,im2_ind_col);

//                int average = (pixel1[0]+pixel1[1]+pixel1[2])/3;
//                std::cout<< average << std::endl;
                if(!(pixel1[0]==0&&pixel1[1]==0&&pixel1[2]==0)) //!(pixel1[0]<255&&pixel1[1]<255&&pixel1[2]<255)
                {
                    double d1_sq = j - centroid1_x, d2_sq = j - centroid2_x, temp = i - centroid1_y;
                    d1_sq = d1_sq*d1_sq + temp*temp; temp = i - centroid2_y;
                    d2_sq = d2_sq*d2_sq + temp*temp;
                    double alpha;
                    if(centroid1_x == centroid2_x && centroid1_y == centroid2_y)
                    {
                        alpha = 0.5;
                    } else
                    {
                        double D_sq = A*j+B*i+C;
                        D_sq = D_sq*D_sq/denomin;
                        double D1_sq = d1_sq-D_sq, D2_sq = d2_sq-D_sq;
                        alpha = D2_sq/(D1_sq+D2_sq);
                    }
                    result.at<cv::Vec3b>(i,j) = pixel1*alpha + pixel2*(1-alpha);
                }
                else
                {
                    result.at<cv::Vec3b>(i,j) = pixel2;
                }
            }
            else
            {
                result.at<cv::Vec3b>(i,j) = pixel1;
            }
        }
    }

//    cv::circle(result,cv::Point(centroid1_x,centroid1_y),2,cv::Scalar(0,0,255),-1);
//    cv::circle(result,cv::Point(centroid2_x,centroid2_y),2,cv::Scalar(0,0,255),-1);     // 绘制质心
//    cv::line(result,cv::Point(centroid1_x,centroid1_y),cv::Point(centroid2_x,centroid2_y),cv::Scalar(0,0,255)); // 绘制质心连线
    cv::imshow("result",result);
    cv::imwrite("./Ransac/result.jpg",result);
    cv::waitKey(0);
}

int main() {
    // test_Ransac_line();
    test_Ransac_sift();
    return 0;
}