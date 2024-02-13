//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_RANSAC_H
#define IMAGEPROCESSINGFROMSCRATCH_RANSAC_H
#include <opencv2/opencv.hpp>
#include <vector>

/////////////////////////ransac///////////////////////////////////////////////////////
template <class T>
std::vector<double> ransac(std::vector<unsigned long> & inners,const std::vector<T> & Data,std::vector<unsigned long> (*SamplingFunc) (const std::vector<T> &),
                           std::vector<double> (*ModelBuildingFunc) (std::vector<unsigned long> &,const std::vector<T> &,const std::vector<unsigned long> &,const double &),
                           const std::vector<double> & last_model,const std::vector<unsigned long> & last_inners, const double & tolerance)
{
    std::vector<unsigned long> cur_inners;
    std::vector<unsigned long> sample = (*SamplingFunc)(Data);
    std::vector<double> cur_model = (*ModelBuildingFunc)(cur_inners,Data,sample,tolerance);
    if(cur_inners.size() > last_inners.size())
    {
        inners = cur_inners;
        return cur_model;
    }
    else
    {
        inners = last_inners;
        return last_model;
    }
}

/////////////////////////////Line/////////////////////////////////////////////////////
struct point2D{
    double x,y;
};

std::vector<unsigned long> lineSample (const std::vector<point2D> & Data)
{
    unsigned long L = Data.size();
    std::vector<unsigned long> sample;
    sample.push_back(rand()%L);
    sample.push_back(rand()%L);
    return sample;
}

std::vector<unsigned long> lineCalcInners(const std::vector<point2D> & Data,
        const std::vector<double> & model,const double & err)
{
    std::vector<unsigned long> inners;
    if(model.size() == 1)
    {
        for(unsigned long i = 0;i < Data.size();i++)
        {
            double D_sq = Data[i].x - model[0];
            D_sq = D_sq * D_sq;
            if(D_sq <= err)
            {
                inners.push_back(i);
            }
        }
    }
    else if(model.size() == 2)
    {
        double denominator = 1 + model[0]*model[0];
        for(unsigned long i = 0;i < Data.size();i++)
        {
            double D_sq = Data[i].y - model[0]*Data[i].x - model[1];
            D_sq = D_sq * D_sq;
            if(D_sq/denominator <= err)
            {
                inners.push_back(i);
            }
        }
    }
    return inners;
}

std::vector<double> lineBuildingFunc (std::vector<unsigned long> & inners,
        const std::vector<point2D> & Data,const std::vector<unsigned long> & sample,const double & err)
{
    std::vector<double> model;
    double x_delta = Data[sample[0]].x - Data[sample[1]].x;
    if(x_delta == 0)
    {
        model.push_back(Data[sample[0]].x);      // x = b
    } else
    {
        double y_delta = Data[sample[0]].y - Data[sample[1]].y;
        double b_delta = Data[sample[0]].x*Data[sample[1]].y-Data[sample[1]].x*Data[sample[0]].y;
        model.push_back(y_delta/x_delta);
        model.push_back(b_delta/x_delta);       // y = kx + b
    }

    inners = lineCalcInners(Data,model,err);

    return model;
}

std::vector<double> LSQ(const std::vector<unsigned long> & inners,const std::vector<point2D> & Data)
{
    std::vector<double> model;
    double xy_ = 0,y_= 0,xsq_ = 0,x_ = 0;
    for(unsigned long i = 0;i < inners.size();i++)
    {
        x_ += Data[inners[i]].x;
        xsq_ += Data[inners[i]].x * Data[inners[i]].x;
        y_ += Data[inners[i]].y;
        xy_ += Data[inners[i]].x * Data[inners[i]].y;
    }
    if(inners.size() > 0)
    {
        xy_ /= inners.size();
        y_ /= inners.size();
        xsq_ /= inners.size();
        x_ /= inners.size();
    }

    double denominator = xsq_ - x_*x_;
    if(denominator == 0)     // this will only be true when all Xs are equal.
    {
        model.push_back(x_);
    } else
    {
        double k = (xy_-x_*y_)/denominator;
        double b = y_ - k * x_;
        model.push_back(k);
        model.push_back(b);
    }
    return model;
}


std::vector<double> Ransac_line(const std::vector<point2D> & Data, bool show = false, double p = 1.0/6, double P = 0.95,
                                 const double & tolerance = 0.03, const unsigned long & sample_num = 2)
/**
 *
 * @param inners : 内点下标集
 * @param Data : 数据集
 * @param Sample_num : 计算模型参数所需样本数
 * @param tolerance : 误差
 * @param p : SNR
 * @param P : 置信度
 * @return : 模型参数集
 */
{
    srand((unsigned)time(NULL));
    assert(Data.size() >= sample_num);
    unsigned long N = (unsigned long)ceil(log(1-P)/log(1-pow(p,sample_num)));

    std::vector<unsigned long> inners;
    std::vector<double> cur_model;
    std::vector<unsigned long> cur_inners;


    double xmax = Data[0].x,xmin = Data[0].x, ymax = Data[0].y, ymin = Data[0].y;
    for(unsigned long i = 1;i < Data.size();i++)
    {
        xmax = Data[i].x > xmax ? Data[i].x : xmax;
        xmin = Data[i].x < xmin ? Data[i].x : xmin;
        ymax = Data[i].y > ymax ? Data[i].y : ymax;
        ymin = Data[i].y < ymin ? Data[i].y : ymin;
    }
    int Xmax = ceil(xmax+10);
    int Xmin = ceil(xmin-10);
    int Ymax = ceil(ymax+10);
    int Ymin = ceil(ymin-10);

    int w = (Xmax - Xmin + 1), h = (Ymax - Ymin + 1);
    int W = 1000;  // 令图片宽度恒为1000
    double zoom_ratio = (1000.0/w);

    cv::Mat img(cv::Size(W, int(ceil(h*zoom_ratio)) ),CV_8UC3);
    img.setTo(255);

    // draw points and axes
    // 注意opencv图像坐标系和xy坐标系的转换
    // 公式: x_ = x - Xmin ; y_ = Ymax - y
    int origin_x = -Xmin, origin_y = Ymax;
    int X_ = Xmax - Xmin, Y_ = Ymax - Ymin;
    if(0 < origin_x && origin_x < X_)
    {
        cv::Point ay1(origin_x*zoom_ratio,0), ay2(origin_x*zoom_ratio,Y_*zoom_ratio);
        cv::line(img,ay1,ay2,cv::Scalar(0,0,0));
    }
    if(0 < origin_y && origin_y < Y_)
    {
        cv::Point ax1(0,origin_y*zoom_ratio), ax2(X_*zoom_ratio,origin_y*zoom_ratio);
        cv::line(img,ax1,ax2,cv::Scalar(0,0,0));
    }
    for(unsigned long i = 0;i < Data.size();i++)
    {
        cv::circle(img,cv::Point((Data[i].x - Xmin)*zoom_ratio,(Ymax - Data[i].y)*zoom_ratio),2,cv::Scalar(255,0,0),-1);
    }

    if(show)
    {
        cv::imshow("current model",img);
        cv::waitKey(500);
    }



    for(unsigned long i = 0;i < N;i++)
    {
        cur_model = ransac<point2D>(cur_inners,Data,lineSample,lineBuildingFunc,cur_model,cur_inners,tolerance);

        if(show)
        {
            cv::Mat cur_img = img.clone();
            // line
            if(cur_model.size() == 1)
            {
                cv::Point p1((cur_model[0]-Xmin)*zoom_ratio,0), p2((cur_model[0]-Xmin)*zoom_ratio,(Ymax-Ymin)*zoom_ratio);
                cv::line(cur_img,p1,p2,cv::Scalar(0,0,255));
            } else
            {
                double p1_x = (Ymax-cur_model[1])/cur_model[0] - Xmin, p1_y = 0;
                double p2_x = (Ymin-cur_model[1])/cur_model[0] - Xmin, p2_y = Ymax -Ymin;
                cv::Point p1(p1_x*zoom_ratio,p1_y*zoom_ratio), p2(p2_x*zoom_ratio,p2_y*zoom_ratio);
                cv::line(cur_img,p1,p2,cv::Scalar(0,0,255));
            }
            // inner points
            for(unsigned long j = 0;j < cur_inners.size();j++)
            {
                cv::circle(cur_img,cv::Point((Data[cur_inners[j]].x - Xmin)*zoom_ratio,(Ymax - Data[cur_inners[j]].y)*zoom_ratio),2,cv::Scalar(0,0,255),-1);
            }
            cv::imshow("current model",cur_img);
            cv::waitKey(500);
        }
    }

    /* optimizer */
    if(!cur_inners.empty())
    {
        cur_model = LSQ(cur_inners,Data);
        cur_inners = lineCalcInners(Data,cur_model,tolerance);

        if(show)
        {
            cv::Mat cur_img = img.clone();
            // line
            if(cur_model.size() == 1)
            {
                cv::Point p1((cur_model[0]-Xmin)*zoom_ratio,0), p2((cur_model[0]-Xmin)*zoom_ratio,(Ymax-Ymin)*zoom_ratio);
                cv::line(cur_img,p1,p2,cv::Scalar(0,0,255));
            } else
            {
                double p1_x = (Ymax-cur_model[1])/cur_model[0] - Xmin, p1_y = 0;
                double p2_x = (Ymin-cur_model[1])/cur_model[0] - Xmin, p2_y = Ymax -Ymin;
                cv::Point p1(p1_x*zoom_ratio,p1_y*zoom_ratio), p2(p2_x*zoom_ratio,p2_y*zoom_ratio);
                cv::line(cur_img,p1,p2,cv::Scalar(0,0,255));
            }
            // inner points
            for(unsigned long j = 0;j < cur_inners.size();j++)
            {
                cv::circle(cur_img,cv::Point((Data[cur_inners[j]].x - Xmin)*zoom_ratio,(Ymax - Data[cur_inners[j]].y)*zoom_ratio),2,cv::Scalar(0,0,255),-1);
            }
            cv::imshow("current model",cur_img);
            cv::waitKey(500);
        }
    }

    return cur_model;
}

//////////////////////////Sift////////////////////////////////////////////////////////
struct sift_pair{
    double x1,y1,x2,y2;
};

std::vector<unsigned long> siftSample(const std::vector<sift_pair> & Data)
{
    unsigned long L = Data.size();
    std::vector<unsigned long> sample;
    sample.push_back(rand()%L);
    sample.push_back(rand()%L);
    sample.push_back(rand()%L);
    sample.push_back(rand()%L);

    return sample;
}

std::vector<unsigned long> siftCalcInners(const std::vector<sift_pair> & Data,
                                          const std::vector<double> & model,const double & err)
{
    std::vector<unsigned long> inners;
    for(unsigned long i = 0;i < Data.size();i++)
    {
        double denominator = model[6]*Data[i].x1 + model[7]*Data[i].y1 + 1;
        double numerator_x = model[0]*Data[i].x1 + model[1]*Data[i].y1 + model[2];
        double numerator_y = model[3]*Data[i].x1 + model[4]*Data[i].y1 + model[5];
        double dx = denominator*Data[i].x2 - numerator_x, dy = denominator*Data[i].y2 - numerator_y;
        if(dx*dx + dy*dy < err)
        {
            inners.push_back(i);
        }
    }

    return inners;
}

std::vector<double> siftBuildingFunc (std::vector<unsigned long> & inners,
                                      const std::vector<sift_pair> & Data,const std::vector<unsigned long> & sample,const double & err)
{
    std::vector<double> model;
    cv::Mat A;
    for(unsigned long i = 0;i < sample.size();i++)
    {
        cv::Mat temp = (cv::Mat_<double >(2,9) <<
                Data[sample[i]].x1,Data[sample[i]].y1,1,0,0,0,-Data[sample[i]].x1*Data[sample[i]].x2,-Data[sample[i]].x2*Data[sample[i]].y1,-Data[sample[i]].x2,
                0,0,0,Data[sample[i]].x1,Data[sample[i]].y1,1,-Data[sample[i]].x1*Data[sample[i]].y2,-Data[sample[i]].y1*Data[sample[i]].y2,-Data[sample[i]].y2);
        A.push_back(temp);
    }

    cv::Mat W,U,Vt;
    cv::SVDecomp(A,W,U,Vt,cv::SVD::FULL_UV);
    int Rows = Vt.rows;
    double ratio = 1/Vt.at<double >(Rows-1,8);

    for(unsigned long i = 0;i < 8;i++)
    {
        model.push_back(Vt.at<double >(Rows-1,i)*ratio);
    }

    inners = siftCalcInners(Data,model,err);
    return model;
}

std::vector<double> siftModelOptimizer(const std::vector<unsigned long> & inners,const std::vector<sift_pair> & Data)
{
    std::vector<double> model;
    cv::Mat A;
    for(unsigned long i = 0;i < inners.size();i++)
    {
        cv::Mat temp = (cv::Mat_<double >(2,9) <<
                Data[inners[i]].x1,Data[inners[i]].y1,1,0,0,0,-Data[inners[i]].x1*Data[inners[i]].x2,-Data[inners[i]].x2*Data[inners[i]].y1,-Data[inners[i]].x2,
                0,0,0,Data[inners[i]].x1,Data[inners[i]].y1,1,-Data[inners[i]].x1*Data[inners[i]].y2,-Data[inners[i]].y1*Data[inners[i]].y2,-Data[inners[i]].y2);
        A.push_back(temp);
    }

    cv::Mat W,U,Vt;
    cv::SVDecomp(A,W,U,Vt,cv::SVD::FULL_UV);
    int Rows = Vt.rows;
    double ratio = 1/Vt.at<double >(Rows-1,8);

    for(unsigned long i = 0;i < 8;i++)
    {
        model.push_back(Vt.at<double >(Rows-1,i)*ratio);
    }

    return model;
}


std::vector<double> Ransac_sift(const std::vector<sift_pair> & Data, double p = 1.0/6, double P = 0.95,
                                const double & tolerance = 3, const unsigned long & sample_num = 4)
{
    srand((unsigned)time(NULL));
    assert(Data.size() >= sample_num);
    unsigned long N = (unsigned long)ceil(log(1-P)/log(1-pow(p,sample_num)));
    std::vector<unsigned long> inners;
    std::vector<double> cur_model;
    std::vector<unsigned long> cur_inners;

    for(unsigned long i = 0;i < N;i++)
    {
        cur_model = ransac<sift_pair>(cur_inners,Data,siftSample,siftBuildingFunc,cur_model,cur_inners,tolerance);
    }

    /* optimizer */
//    if(!cur_inners.empty())
//    {
//        cur_model = siftModelOptimizer(cur_inners,Data);
//        cur_inners = siftCalcInners(Data,cur_model,tolerance);
//    }

    return cur_model;
}


#endif //IMAGEPROCESSINGFROMSCRATCH_RANSAC_H