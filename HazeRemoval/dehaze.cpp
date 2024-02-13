//
// author: 会飞的吴克
//

#include "dehaze.h"
#include "fastguidedfilter.h"

namespace DarkChannel{

    cv::Mat dehaze(const cv::Mat &img) {
        __Assert__(img.type() == CV_8UC3,"3 channel images only.");
        cv::Mat img_double;
        img.convertTo(img_double,CV_64FC3);
        img_double /= 255;

        // get dark channel
        int patch_radius = 7;
        cv::Mat darkchannel = cv::Mat::zeros(img_double.rows,img_double.cols,CV_64F);
        for(int i = 0;i < darkchannel.rows;i++)
        {
            for(int j = 0;j < darkchannel.cols;j++)
            {
                int r_start = std::max(0,i-patch_radius);
                int r_end = std::min(darkchannel.rows,i+patch_radius+1);
                int c_start = std::max(0,j-patch_radius);
                int c_end = std::min(darkchannel.cols,j+patch_radius+1);
                double dark = 255;
                for(int r = r_start;r < r_end;r++)
                {
                    for(int c = c_start;c < c_end;c++)
                    {
                        const cv::Vec3d& val = img_double.at<cv::Vec3d>(r,c);
                        dark = std::min(dark,val[0]);
                        dark = std::min(dark,val[1]);
                        dark = std::min(dark,val[2]);
                    }
                }
                darkchannel.at<double>(i,j) = dark;
            }
        }


        // get atmospheric light
        int pixels = img_double.rows * img_double.cols;
        int num = pixels/1000; // 0.1%
        std::vector<std::pair<double,int> > V;
        V.reserve(pixels);
        int k = 0;
        for(int i = 0;i < darkchannel.rows;i++)
        {
            for(int j = 0;j < darkchannel.cols;j++)
            {
                V.emplace_back(darkchannel.at<double >(i,j),k);
                k++;
            }
        }
        std::sort(V.begin(),V.end(),[](const std::pair<double,int>& p1,const std::pair<double,int>& p2){ return p1.first > p2.first;});

        double atmospheric_light[] = {0,0,0};
        for(k = 0;k < num;k++)
        {
            int r = V[k].second/darkchannel.cols;
            int c = V[k].second%darkchannel.cols;
            const cv::Vec3d& val = img_double.at<cv::Vec3d>(r,c);
            atmospheric_light[0] += val[0];
            atmospheric_light[1] += val[1];
            atmospheric_light[2] += val[2];
        }
        atmospheric_light[0] /= num;
        atmospheric_light[1] /= num;
        atmospheric_light[2] /= num;


        double omega = 0.95;
        cv::Mat channels[3];
        cv::split(img_double,channels);
        for(k = 0;k < 3;k++)
        {
            channels[k] /= atmospheric_light[k];
        }
        cv::Mat temp;
        cv::merge(channels,3,temp);
        // get dark channel of temp.
        cv::Mat temp_dark = cv::Mat::zeros(temp.rows,temp.cols,CV_64F);
        for(int i = 0;i < temp.rows;i++)
        {
            for(int j = 0;j < temp.cols;j++)
            {
                int r_start = std::max(0,i-patch_radius);
                int r_end = std::min(temp.rows,i+patch_radius+1);
                int c_start = std::max(0,j-patch_radius);
                int c_end = std::min(temp.cols,j+patch_radius+1);
                double dark = 255;
                for(int r = r_start;r < r_end;r++)
                {
                    for(int c = c_start;c < c_end;c++)
                    {
                        const cv::Vec3d& val = temp.at<cv::Vec3d>(r,c);
                        dark = std::min(dark,val[0]);
                        dark = std::min(dark,val[1]);
                        dark = std::min(dark,val[2]);
                    }
                }
                temp_dark.at<double>(i,j) = dark;
            }
        }
        // get transmission.
        cv::Mat transmission = 1-omega*temp_dark;
        transmission = fastGuidedFilter(img_double,transmission,40,0.1,5);

        double t0 = 0.1;
        cv::split(img_double,channels);
        cv::Mat trans_;
        cv::max(transmission,t0,trans_);
        cv::Mat temp_;
        for(k = 0;k < 3;k++)
        {
            cv::divide((channels[k]-atmospheric_light[k]),trans_,temp_);
            channels[k] = atmospheric_light[k] + temp_;
        }

        cv::Mat res;
        cv::merge(channels,3,res);
        double maxv;
        minMaxLoc(res,nullptr,&maxv,nullptr,nullptr);
        res /= maxv;
        res *= 255;
        res.convertTo(res,CV_8UC3);

        return res;
    }
}
