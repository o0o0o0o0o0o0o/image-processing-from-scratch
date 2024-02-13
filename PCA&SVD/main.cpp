//
// author: 会飞的吴克
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "PCA.h"
#include "svd.h"

void test_pca()
{
    cv::Mat img = cv::imread("./PCA&SVD/0.jpg");
    cv::imshow("img",img);
    cv::waitKey(0);

    std::vector<std::vector<double > > data[img.channels()];
    for(unsigned long i = 0;i < img.rows;i++)
    {
        std::vector<double > c[img.channels()];
        for(unsigned long j = 0;j < img.cols;j++)
        {
            for(unsigned long k = 0;k < img.channels();k++)
            {
                c[k].push_back(img.at<cv::Vec3b>(i,j)[k]);
            }
        }
        for(unsigned long k = 0;k < img.channels();k++)
        {
            data[k].push_back(c[k]);
        }
    } // img channels convert to vector<vector<>>


    std::pair<std::pair<std::vector<double >, std::vector<std::vector<double > > >,std::vector<std::vector<double > > > PC[img.channels()];
    std::vector<std::vector<double > > recover_data[img.channels()];
    for(unsigned long k = 0;k < img.channels();k++)
    {
        PC[k] = EigenValue::PCA(data[k],0,.99);
        recover_data[k] = EigenValue::multiply(PC[k].second,PC[k].first.second);
        std::cout<<PC[k].first.second.size()<<std::endl;
    } // compute PCA for different channels


    cv::Mat res(cv::Size(img.cols, img.rows ),CV_8UC3);
    for(unsigned long k = 0;k < img.channels();k++)
    {
        for(unsigned long i = 0;i < recover_data[k].size();i++)
        {
            for(unsigned long j = 0;j < recover_data[k][i].size();j++)
            {
                res.at<cv::Vec3b >(i,j)[k] =  recover_data[k][i][j] + PC[k].first.first[j];
            }
        }
    }

    cv::imwrite("./PCA&SVD/0_.jpg",res);
    cv::imshow("img",res);
    cv::waitKey(0);
}

void test_svd()
{
    std::vector<std::vector<double> > A ={{1,2,3},{3,4,5},{5,6,7},{7,8,9}};
    auto result = EigenValue::SVD(A);
    for(unsigned long i = 0;i < result.first.size();i++)
    {
        std::cout << result.first[i] <<" ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    for(unsigned long i = 0;i < result.second.first.size();i++)
    {
        for(unsigned long j = 0;j < result.second.first[i].size();j++)
        {
            std::cout<< result.second.first[i][j] << " ";
        }
        std::cout<<std::endl;
    }
    std::cout << std::endl;

    for(unsigned long i = 0;i < result.second.second.size();i++)
    {
        for(unsigned long j = 0;j < result.second.second[i].size();j++)
        {
            std::cout<< result.second.second[i][j] << " ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    std::vector<std::vector<double> > sigma;
    for(unsigned long i = 0;i < result.first.size();i++)
    {
        std::vector<double> temp;
        for(unsigned long j = 0;j < result.first.size();j++)
        {
            if(i == j) temp.push_back(result.first[i]);
            else temp.push_back(0);
        }
        sigma.push_back(temp);
    }
//    auto recover = EigenValue::multiply(result.second.first,sigma);
    auto recover = EigenValue::multiply(result.second.first,EigenValue::multiply(sigma,EigenValue::transpose(result.second.second)));

    for(unsigned long i = 0;i < recover.size();i++)
    {
        for(unsigned long j = 0;j < recover[i].size();j++)
        {
            std::cout<< recover[i][j] << " ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

int main() {
    test_pca();
    // test_svd();
    return 0;
}