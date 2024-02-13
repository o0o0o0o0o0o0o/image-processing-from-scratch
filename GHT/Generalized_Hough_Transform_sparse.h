//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_GENERALIZED_HOUGH_TRANSFORM_SPARSE_H
#define IMAGEPROCESSINGFROMSCRATCH_GENERALIZED_HOUGH_TRANSFORM_SPARSE_H
#include <iostream>
#include <opencv2/opencv.hpp>
namespace GHT_SPARSE{
    const double PI  = 3.14159265358979323846264338327950288;
    struct range{
        double left;
        double right;
    };
    struct Vec2d{
        double x;
        double y;
    };
    struct GHParams{
        int xc;
        int yc;
        int theta;
        int scale;
    };
    struct Color{
        int r,g,b;
    };
    class SparseVec{
    private:
        std::map<unsigned long,int> data;
        unsigned long vectorlength = 0;
    public:
        explicit SparseVec(const unsigned long &n){vectorlength=n;}

        unsigned long getLength()const{ return vectorlength;}

        int get(const unsigned long &i) const
        {
            assert(i < vectorlength);
            auto itr = data.find(i);
            if(itr != data.end())
            {
                return itr->second;
            }
            else
            {
                return 0;
            }
        }

        void set(const unsigned long &i,const int &val)
        {
            assert(i < vectorlength);
            auto itr = data.find(i);
            if(itr != data.end())
            {
                if(val == 0)
                {
                    data.erase(itr);
                }
                else
                {
                    itr->second = val;
                }
            }
            else
            {
                if(val != 0)
                {
                    data.insert(std::make_pair(i,val));
                }
            }
        }

        int increment(const unsigned long &i, const int &val = 1)
        {
            assert(i < vectorlength);
            int res = 0;
            auto itr = data.find(i);
            if(itr != data.end())
            {
                itr->second += val;
                res = itr->second;
                if(itr->second == 0)
                {
                    data.erase(itr);
                }
            }
            else
            {
                if(val != 0)
                {
                    data.insert(std::make_pair(i, val));
                }
                res = val;
            }
            return res; // return the value after increment.
        }

        template <typename T>
        std::map<unsigned long,int>::size_type erase_if(const T &Condition)
        {

            auto old_size = data.size();
            for(auto it = data.begin();it != data.end();)
            {
                if(Condition(it->second)){it = data.erase(it);}
                else{it++;}
            }
            return old_size - data.size();
        }
    };

    typedef std::vector<std::vector<Vec2d> > RTable;

    cv::Mat rotateAndScale(const cv::Mat& image, double angle, double scale)
    /*
     * image: image to be rotate and scaled.
     * angle: the angle you want to rotate
     * (counter clockwise[note that downward is the forward direction for y-aixs in an image, so this counter clockwise will seems to be clockwise in commonsense])
     * scale: scale factor
     *
     * output: image after scale and rotate
     */
    {
        cv::Mat T = cv::Mat::eye(3, 3, CV_32FC1);
        T.at<float>(0, 2) = -image.cols/2.0;
        T.at<float>(1, 2) = -image.rows/2.0;

        cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
        R.at<float>(0, 0) =  cos(angle);
        R.at<float>(0, 1) = -sin(angle);
        R.at<float>(1, 0) =  sin(angle);
        R.at<float>(1, 1) =  cos(angle);

        cv::Mat S = cv::Mat::eye(3, 3, CV_32FC1);
        S.at<float>(0, 0) = scale;
        S.at<float>(1, 1) = scale;

        cv::Mat H = R*S*T;

        cv::Mat corners(1, 4, CV_32FC2);
        corners.at<cv::Vec2f>(0, 0) = cv::Vec2f(0,0);
        corners.at<cv::Vec2f>(0, 1) = cv::Vec2f(0,image.rows);
        corners.at<cv::Vec2f>(0, 2) = cv::Vec2f(image.cols,0);
        corners.at<cv::Vec2f>(0, 3) = cv::Vec2f(image.cols,image.rows);
        perspectiveTransform(corners, corners, H);

        float x_start = std::min( std::min( corners.at<cv::Vec2f>(0, 0)[0], corners.at<cv::Vec2f>(0, 1)[0]),
                                  std::min( corners.at<cv::Vec2f>(0, 2)[0], corners.at<cv::Vec2f>(0, 3)[0]) );
        float x_end   = std::max( std::max( corners.at<cv::Vec2f>(0, 0)[0], corners.at<cv::Vec2f>(0, 1)[0]),
                                  std::max( corners.at<cv::Vec2f>(0, 2)[0], corners.at<cv::Vec2f>(0, 3)[0]) );
        float y_start = std::min( std::min( corners.at<cv::Vec2f>(0, 0)[1], corners.at<cv::Vec2f>(0, 1)[1]),
                                  std::min( corners.at<cv::Vec2f>(0, 2)[1], corners.at<cv::Vec2f>(0, 3)[1]) );
        float y_end   = std::max( std::max( corners.at<cv::Vec2f>(0, 0)[1], corners.at<cv::Vec2f>(0, 1)[1]),
                                  std::max( corners.at<cv::Vec2f>(0, 2)[1], corners.at<cv::Vec2f>(0, 3)[1]) );

        T.at<float>(0, 0) = 1;
        T.at<float>(1, 1) = 1;
        T.at<float>(2, 2) = 1;
        T.at<float>(0, 2) = (x_end - x_start + 1)/2.0;
        T.at<float>(1, 2) = (y_end - y_start + 1)/2.0;

        H = T * H;
        cv::Mat output;
        warpPerspective(image, output, H, cv::Size(x_end - x_start + 1, y_end - y_start + 1), cv::INTER_LINEAR);

        return output;
    }

    cv::Mat visualize(const cv::Mat& templateImage, const cv::Mat& queryImage, const std::vector<GHParams >& objList,
                      const range &scale_range, const int &scale_steps, const range &angle_range, const int & angle_steps,const Color &color)
                      /*
                       * templateImage: edges of template(1 channel binary image)
                       * queryImage: image to detect object.
                       * objList: parameters of detected objects.
                       * color: the color you want to mark the detected objects.(rgb)
                       */
    {
        assert(templateImage.channels() == 1 && queryImage.channels() == 3);

        cv::Mat queryImage_temp;
        queryImage.convertTo(queryImage_temp,CV_32F);
        std::vector<cv::Mat> channels;
        cv::split(queryImage_temp,channels);

        double scale_step = (scale_range.right - scale_range.left)/scale_steps;
        double angle_step = (angle_range.right - angle_range.left)/angle_steps;
        double scale, angle;
        for(auto it = objList.begin(); it != objList.end(); it++)
        {
            scale = scale_step*((*it).scale + 0.5) + scale_range.left;
            angle = angle_step*((*it).theta + 0.5) + angle_range.left;

//            std::cout<< angle << std::endl;

            cv::Mat binMask = rotateAndScale(templateImage, angle, scale);
//            cv::imshow("Mask",binMask);
//            cv::waitKey(0);
            binMask.convertTo(binMask,CV_32F);

            cv::Rect binArea = cv::Rect(0, 0, binMask.cols, binMask.rows);
            cv::Rect imgArea = cv::Rect((*it).xc-binMask.cols/2., (*it).yc-binMask.rows/2., binMask.cols, binMask.rows);
            if ( imgArea.x < 0 )
            {
                binArea.x = abs( imgArea.x );
                binArea.width = binMask.cols - binArea.x;
                imgArea.x = 0;
                imgArea.width = binArea.width;
            }
            if ( imgArea.y < 0 )
            {
                binArea.y = abs( imgArea.y );
                binArea.height = binMask.rows - binArea.y;
                imgArea.y = 0;
                imgArea.height = binArea.height;
            }
            if ( imgArea.x + imgArea.width >= queryImage_temp.cols )
            {
                binArea.width = queryImage_temp.cols - imgArea.x;
                imgArea.width = binArea.width;
            }
            if ( imgArea.y + imgArea.height >= queryImage_temp.rows )
            {
                binArea.height = queryImage_temp.rows - imgArea.y;
                imgArea.height = binArea.height;
            }

            cv::Mat binRoi = cv::Mat(binMask, binArea);
            cv::Mat tmp = binRoi.clone();
            binMask = 1 - binMask/255;

            cv::Mat imgRoi;

            imgRoi = cv::Mat(channels[0], imgArea);
            multiply(imgRoi, binRoi, imgRoi);
            imgRoi = imgRoi + tmp*color.b/255;

            imgRoi = cv::Mat(channels[1], imgArea);
            multiply(imgRoi, binRoi, imgRoi);
            imgRoi = imgRoi + tmp*color.g/255;

            imgRoi = cv::Mat(channels[2], imgArea);
            multiply(imgRoi, binRoi, imgRoi);
            imgRoi = imgRoi + tmp*color.r/255;
        }

        cv::Mat display;
        merge(channels, display);
        display.convertTo(display,CV_8UC1);
        return display;
    }

    RTable buildRTable(const cv::Mat &template_image, int phi_steps)
    /*
     * template_image: template image
     * phi_steps: number of angles that you wanna record.
     *
     * return: R-Table
     */
    {
        assert(phi_steps > 0);
        RTable RT;
        RT.reserve(phi_steps);
        double step = 2*PI / phi_steps;
        for(int i = 0;i < phi_steps;i++)
        {
            RT.push_back(std::vector<Vec2d>());
        }

        cv::Mat edges;
        cv::cvtColor(template_image,edges,cv::COLOR_BGR2GRAY);
        cv::Mat x_grad, y_grad;
        cv::Sobel(edges,x_grad,CV_64FC1,1,0);
        cv::Sobel(edges,y_grad,CV_64FC1,0,1);
//        cv::blur(x_grad,x_grad,cv::Size(3,3));
//        cv::blur(y_grad,y_grad,cv::Size(3,3));
        cv::Canny(edges,edges,50,150);

        double center_x = edges.cols / 2.0, center_y = edges.rows / 2.0;
        for(int i = 0;i < edges.rows;i++)
        {
            const uchar *edges_p = edges.ptr<uchar>(i);
            double *x_grad_p = x_grad.ptr<double >(i);
            double *y_grad_p = y_grad.ptr<double >(i);
            for(int j = 0;j < edges.cols;j++)
            {
                double dx = x_grad_p[j], dy = y_grad_p[j];
                if(edges_p[j] && (abs(dx) > 0 || abs(dy) > 0))
                {
                    double phi = atan2(dy,dx);
                    double temp = (phi + PI)/step;
                    temp = temp > 0 ? temp : 0;
                    temp = temp < phi_steps ? temp : phi_steps - 1;
                    int ind = int(floor(temp));

                    RT[ind].push_back({j-center_x,i-center_y});
                }
            }
        }

        return RT;
    }

    SparseVec vote(const cv::Mat &src, const RTable &RT, const range &scale_range,
                   const int &scale_steps, const range &angle_range,
                   const int & angle_steps, const double &threshold_factor,
                   const int &image_steps = 2)
    /*
     * src: query image
     * RT: R-Table
     * scale_range: range of s
     * scale_steps: number of scales that you wanna check.
     * angle_range: range of theta
     * angle_steps: number of angles that you wanna check.
     * image_steps: down sample interval.
     */
    {
        assert(image_steps >= 1);
        assert(threshold_factor > 0 && threshold_factor < 1);

        cv::Mat edges;
        cv::cvtColor(src,edges,cv::COLOR_BGR2GRAY);

        double T = 0.9 * threshold_factor; // early threshold factor.
        unsigned long angle_dim = scale_steps;
        unsigned long y_dim = angle_steps*angle_dim;
        unsigned long x_dim = edges.rows*y_dim;
        SparseVec res(edges.cols*x_dim);

        double scale_step = (scale_range.right-scale_range.left)/scale_steps;
        double angle_step = (angle_range.right-angle_range.left)/angle_steps;
        double  two_PI = 2*PI;
        double phi_step = two_PI/RT.size();

        cv::Mat x_grad, y_grad;
        cv::Sobel(edges,x_grad,CV_64FC1,1,0);
        cv::Sobel(edges,y_grad,CV_64FC1,0,1);
        cv::Canny(edges,edges,50,150);
//        cv::blur(x_grad,x_grad,cv::Size(3,3));
//        cv::blur(y_grad,y_grad,cv::Size(3,3));

        std::cout << "voting" <<std::endl;
        int voters = 0;
        int maximum = 0;
        int cur_val = 0;
        for(int image_step_i = 0;image_step_i < image_steps;image_step_i++)
        {
            for(int image_step_j = 0;image_step_j < image_steps;image_step_j++)
            {
                for(int i = image_step_i;i < edges.rows;i+=image_steps)
                {
                    const uchar *edges_p = edges.ptr<uchar>(i);
                    double *x_grad_p = x_grad.ptr<double >(i);
                    double *y_grad_p = y_grad.ptr<double >(i);
                    for(int j = image_step_j;j < edges.cols;j+=image_steps)
                    {
                        double dx = x_grad_p[j], dy = y_grad_p[j];
                        if(edges_p[j] && (abs(dx) > 0 || abs(dy) > 0))
                        {
//                            std::cout << dx << " " << dy << std::endl;
                            double phi = atan2(dy,dx);
                            for(int theta_ind = 0;theta_ind < angle_steps;theta_ind++)
                            {
                                double theta = angle_range.left + ((theta_ind+0.5)*angle_step);
                                double Cos = cos(theta), Sin = sin(theta);

//                                double dx_ = dx * Cos + dy * Sin;
//                                double dy_ = -dx * Sin + dy * Cos;
//                                double new_phi = atan2(dy_,dx_);
                                double new_phi = phi - theta;
                                new_phi -= ceil(new_phi/two_PI)*two_PI;
                                new_phi = new_phi <= -PI ? new_phi + two_PI : new_phi;
                                double temp = (new_phi + PI)/phi_step;
                                temp = temp > 0 ? temp : 0;
                                temp = temp < RT.size() ? temp : RT.size() - 1;
                                int ind = int(floor(temp));

                                auto displacement_vecs = RT[ind];
                                for(auto vec:displacement_vecs)
                                {
                                    double x_ = vec.x * Cos - vec.y * Sin;
                                    double y_ = vec.x * Sin + vec.y * Cos;

                                    for(int scale_ind = 0;scale_ind < scale_steps;scale_ind++)
                                    {
                                        double s = scale_range.left + ((scale_ind+0.5)*scale_step);
                                        double xc = j - x_*s, yc = i - y_*s;
                                        if(xc >= 0 && xc < edges.cols && yc >= 0 && yc < edges.rows)
                                        {
                                            int x_ind = int(xc), y_ind = int(yc);
                                            cur_val = res.increment(x_ind*x_dim+y_ind*y_dim+theta_ind*angle_dim+scale_ind);
                                            if(cur_val > maximum)
                                            {
                                                maximum = cur_val;
                                            }
                                        }
                                    }
                                }
                            }
                            voters++;
                            std::cout << "voter " << voters <<std::endl;
                        }
                    }
                }
                double thres = T * maximum;
                res.erase_if(
                        [&thres](int &val){
                            return val < thres;
                        });
                //early thresholding.
            }
        }
        std::cout << "done" <<std::endl;

        return res;
    }

    std::vector<GHParams > NMSandThresholding4D(const SparseVec &houghspace, const GHParams &dims, const double &threshold_factor,const GHParams &NMS_raidus)
    {
        unsigned long angle_dim = dims.scale;
        unsigned long y_dim = dims.theta*angle_dim;
        unsigned long x_dim = dims.yc*y_dim;
        assert(houghspace.getLength() == (x_dim*dims.xc));

        std::vector<GHParams > extremes;
        int maximum = 0;
        for(int i = 0;i < dims.xc;i++)
        {
            for(int j = 0;j < dims.yc;j++)
            {
                for(int k = 0;k < dims.theta;k++)
                {
                    for(int l = 0;l < dims.scale;l++)
                    {
                        int val = houghspace.get(i*x_dim+j*y_dim+k*angle_dim+l);
                        if(val > maximum){maximum=val;}
                        if(val != 0) // 0 is not extreme, since votes are positive.
                        {
                            int start_x = i - NMS_raidus.xc, end_x = i + NMS_raidus.xc + 1;
                            int start_y = j - NMS_raidus.yc, end_y = j + NMS_raidus.yc + 1;
                            int start_theta = k - NMS_raidus.theta, end_theta = k + NMS_raidus.theta + 1;
                            int start_scale = l - NMS_raidus.scale, end_scale = l + NMS_raidus.scale + 1;
                            start_x = start_x > 0 ? start_x : 0;
                            end_x = end_x < dims.xc ? end_x : dims.xc;
                            start_y = start_y > 0 ? start_y : 0;
                            end_y = end_y < dims.yc ? end_y : dims.yc;
                            start_theta = start_theta > 0 ? start_theta : 0;
                            end_theta = end_theta < dims.theta ? end_theta : dims.theta;
                            start_scale = start_scale > 0 ? start_scale : 0;
                            end_scale = end_scale < dims.scale ? end_scale : dims.scale;

                            bool isExtreme = true;
                            for(int m = start_x;m < end_x;m++)
                            {
                                for(int n = start_y;n < end_y;n++)
                                {
                                    for(int o = start_theta;o < end_theta;o++)
                                    {
                                        for(int p = start_scale;p < end_scale;p++)
                                        {
                                            isExtreme = isExtreme && (val >= houghspace.get(m*x_dim+n*y_dim+o*angle_dim+p));
                                            if(!isExtreme){ break;}
                                        }
                                        if(!isExtreme){ break;}
                                    }
                                    if(!isExtreme){ break;}
                                }
                                if(!isExtreme){ break;}
                            }

                            if(isExtreme)
                            {
                                extremes.push_back({i,j,k,l});
                            }
                        }
                    }
                }
            }
        }

        double thresh = threshold_factor * maximum;
        extremes.erase(
                std::remove_if(extremes.begin(),extremes.end(),
                               [&houghspace,&thresh,&x_dim,&y_dim,&angle_dim]
                                       (GHParams &e)
                               { return houghspace.get(e.xc*x_dim+e.yc*y_dim+e.theta*angle_dim+e.scale) < thresh;}),
                extremes.end()
        );
        return extremes;
    }

    std::vector<GHParams > getParameters(const SparseVec &houghspace, const GHParams &dims,const double &threshold_factor, const GHParams &NMS_raidus)
    {
        return NMSandThresholding4D(houghspace,dims,threshold_factor,NMS_raidus);
    }

    cv::Mat generalhoughtransform(const cv::Mat &src, const cv::Mat &pattern, const Color &color,const int &image_steps=2,
                                  const GHParams &NMS_raidus={1,1,1,1},
                                  double threshold_factor = 0.8, int phi_steps = 180,
                                  range scale_range = {0.5,5.0}, int scale_steps = 45,
                                  range angle_range = {-PI/2,PI/2}, int angle_steps = 90)
    {
        assert(threshold_factor > 0 && threshold_factor < 1);
        assert(scale_steps > 0 && angle_steps > 0);
        assert(scale_range.left > 0 && scale_range.right > 0);
        if(scale_range.left > scale_range.right)
        {
            double temp = scale_range.right;
            scale_range.right = scale_range.left;
            scale_range.left = temp;
        }

        // get R-Table
        auto RT = buildRTable(pattern, phi_steps);

        // vote
        auto Bins = vote(src,RT,scale_range,scale_steps,angle_range,angle_steps,threshold_factor,image_steps);

        // get parameters
        auto params = getParameters(Bins,{src.cols,src.rows,angle_steps,scale_steps},threshold_factor,NMS_raidus);
        std::cout << params.size() << std::endl;

        // draw detection result.
        cv::Mat pattern_curve;
        cv::Canny(pattern,pattern_curve,50,150);
        auto res = visualize(pattern_curve,src,params,scale_range,scale_steps,angle_range,angle_steps,color);
        return res;
    }
}
#endif //IMAGEPROCESSINGFROMSCRATCH_GENERALIZED_HOUGH_TRANSFORM_SPARSE_H
