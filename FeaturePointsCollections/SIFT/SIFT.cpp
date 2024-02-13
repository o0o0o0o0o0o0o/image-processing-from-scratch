#include "SIFT.h"

namespace SIFT{

    cv::Mat downsampling(const cv::Mat& img,int step)
    {
        assert(img.type() == CV_8U);
        cv::Mat res = cv::Mat::zeros((img.rows-1)/step + 1,(img.cols-1)/step+1,img.type());
        for(int i = 0,k = 0;i < img.rows;i+=step,k++)
        {
            const uchar* img_ptr = img.ptr<uchar>(i);
            uchar* res_ptr = res.ptr<uchar>(k);
            for(int j = 0,l = 0;j < img.cols;j+=step,l++)
            {
                res_ptr[l] = img_ptr[j];
            }
        }
        return res;
    }

    void GetMainDirection(const cv::Mat& img,int r,int c,int radius,float sigma,int BinNum,float& omax, std::vector<float>& hist)
    {
        float expf_scale = -1.0 / (2.0 * sigma * sigma);

        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> W;
        std::vector<float> temphist(BinNum,0);

        int k = 0;
        for(int i = -radius;i < radius+1;i++)
        {
            int y = r + i;
            if(y <= 0 or y >= img.rows - 1)
            { continue; }
            for(int j = -radius;j < radius+1;j++)
            {
                int x = c + j;
                if(x <= 0 or x >= img.cols - 1)
                {continue;}

                X.push_back(img.at<float >(y, x + 1) - img.at<float >(y, x - 1));
                Y.push_back(img.at<float >(y - 1, x) - img.at<float >(y + 1, x));
                W.push_back((i * i + j * j) * expf_scale);
                k++;
            }
        }
        int length = k;
        for(auto& w:W){w = std::expf(w);}

        std::vector<float> Ori(length), Mag(length);
        for(int i = 0;i < length;i++)
        {
            Ori[i] = std::atan2f(Y[i],X[i])*180/M_PI;
            Mag[i] = std::sqrtf(X[i]*X[i]+Y[i]*Y[i]);
        }


        for(k = 0;k < length;k++)
        {
            int bin = int(std::roundf((BinNum / 360.0) * Ori[k]));
            if(bin >= BinNum)
            {bin -= BinNum;}
            if(bin < 0)
            {bin += BinNum;}
            temphist[bin] += W[k] * Mag[k];
        }

        float temp[] = {temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]};
        temphist.insert(temphist.begin(), temp[0]);
        temphist.insert(temphist.begin(), temp[1]);
        temphist.insert(temphist.end(), temp[2]);
        temphist.insert(temphist.end(), temp[3]);

        hist.clear();
        hist.reserve(BinNum);
        for(int i = 0;i < BinNum;i++)
        {
            hist.push_back((temphist[i] + temphist[i+4]) * (1.0 / 16.0) + (temphist[i+1] + temphist[i+3]) * (4.0 / 16.0) + temphist[i+2] * (6.0 / 16.0));
        }

        omax = *std::max_element(hist.begin(),hist.end());
    }

    void calcSIFTDescriptor(const cv::Mat& img,float c,float r,double ori,double scl,int d,int n,double* dst,
            float SIFT_DESCR_SCL_FCTR = 3.0,float SIFT_DESCR_MAG_THR = 0.2,float SIFT_INT_DESCR_FCTR = 512.0,float _FLT_EPSILON = 1.19209290E-07)
    {
        int pt_c = int(std::roundf(c)), pt_r = int(std::roundf(r));
        float cos_t = std::cos(ori * (M_PI / 180));
        float sin_t = std::sin(ori * (M_PI / 180));
        float bins_per_rad = n / 360.0;
        float exp_scale = -1.0 / (d * d * 0.5);
        float hist_width = SIFT_DESCR_SCL_FCTR * scl;
        int radius = int(std::round(hist_width * 1.4142135623730951 * (d + 1) * 0.5));
        cos_t /= hist_width;
        sin_t /= hist_width;

        int rows = img.rows;
        int cols = img.cols;

        std::vector<float> hist(((d+2)*(d+2)*(n+2)),0);
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> RBin;
        std::vector<float> CBin;
        std::vector<float> W;

        float c_rot,r_rot,rbin,cbin;
        int rr,cc;
        int k = 0;
        for(int i = -radius;i < radius+1;i++)
        {
            for(int j = -radius;j < radius+1;j++)
            {
                c_rot = j * cos_t - i * sin_t;
                r_rot = j * sin_t + i * cos_t;
                rbin = r_rot + d / 2 - 0.5;
                cbin = c_rot + d / 2 - 0.5;
                rr = pt_r + i;
                cc = pt_c + j;

                if(rbin > -1 and rbin < d and cbin > -1 and cbin < d and rr > 0 and rr < rows - 1 and cc > 0 and cc < cols - 1)
                {
                    X.push_back(img.at<float >(rr, cc+1) - img.at<float >(rr, cc-1));
                    Y.push_back(img.at<float >(rr-1, cc) - img.at<float >(rr+1, cc));
                    RBin.push_back(rbin);
                    CBin.push_back(cbin);
                    W.push_back((c_rot * c_rot + r_rot * r_rot) * exp_scale);
                    k++;
                }
            }
        }

        int length = k;
        for(auto& w:W){w = std::expf(w);}
        std::vector<float> Ori(length), Mag(length);
        for(int i = 0;i < length;i++)
        {
            Ori[i] = std::atan2f(Y[i],X[i])*180/M_PI;
            Mag[i] = std::sqrtf(X[i]*X[i]+Y[i]*Y[i]);
        }

        float obin,mag;
        for(k = 0;k < length;k++)
        {
            rbin = RBin[k];
            cbin = CBin[k];
            obin = (Ori[k] - ori) * bins_per_rad;
            mag = Mag[k] * W[k];

            int r0 = int(rbin);
            int c0 = int(cbin);
            int o0 = int(obin);
            rbin -= r0;
            cbin -= c0;
            obin -= o0;

            if(o0 < 0){ o0 += n; }
            if(o0 >= n){ o0 -= n; }

            float v_r1 = mag * rbin;
            float v_r0 = mag - v_r1;

            float v_rc11 = v_r1 * cbin;
            float v_rc10 = v_r1 - v_rc11;

            float v_rc01 = v_r0 * cbin;
            float v_rc00 = v_r0 - v_rc01;

            float v_rco111 = v_rc11 * obin;
            float v_rco110 = v_rc11 - v_rco111;

            float v_rco101 = v_rc10 * obin;
            float v_rco100 = v_rc10 - v_rco101;

            float v_rco011 = v_rc01 * obin;
            float v_rco010 = v_rc01 - v_rco011;

            float v_rco001 = v_rc00 * obin;
            float v_rco000 = v_rc00 - v_rco001;

            int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
            hist[idx] += v_rco000;
            hist[idx+1] += v_rco001;
            hist[idx + (n+2)] += v_rco010;
            hist[idx + (n+3)] += v_rco011;
            hist[idx+(d+2) * (n+2)] += v_rco100;
            hist[idx+(d+2) * (n+2)+1] += v_rco101;
            hist[idx+(d+3) * (n+2)] += v_rco110;
            hist[idx+(d+3) * (n+2)+1] += v_rco111;
        }

        int ind = 0;
        for(int i = 0;i < d;i++)
        {
            for(int j = 0;j < d;j++)
            {
                int idx = ((i+1) * (d+2) + (j+1)) * (n+2);
                hist[idx] += hist[idx+n];
                hist[idx+1] += hist[idx+n+1];
                for(k = 0;k < n;k++)
                {
                    dst[ind] = hist[idx+k];
                    ind++;
                }
            }
        }


        double nrm2 = 0;
        length = d * d * n;
        for(k = 0;k < length;k++)
        {
            nrm2 += dst[k] * dst[k];
        }
        double thr = std::sqrt(nrm2) * SIFT_DESCR_MAG_THR;

        nrm2 = 0;
        for(int i = 0;i < length;i++)
        {
            double val = std::min(dst[i], thr);
            dst[i] = val;
            nrm2 += val * val;
        }
        nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), (double)_FLT_EPSILON);
        for(k = 0;k < length;k++)
        {
            dst[k] = std::min(std::max(dst[k] * nrm2,0.),255.);
        }
    }

    void extract(const cv::Mat &img, Keypoints& keypoints) {
        float SIFT_SIGMA = 1.6;
        float SIFT_INIT_SIGMA = 0.5;
        float sigma0 = std::sqrt(SIFT_SIGMA*SIFT_SIGMA-SIFT_INIT_SIGMA*SIFT_INIT_SIGMA);

        cv::Mat grayscale;
        cv::cvtColor(img,grayscale,cv::COLOR_BGR2GRAY);

        // get Guassian Pyramid and DoG
        int n = 3;
        int S = n + 3;
        int O = int(std::log2(std::min(grayscale.rows, grayscale.cols))) - 3;

        float k = std::powf(2,(1.f / n));
        float sigma[O][S];
        for(int i = 0;i < O;i++)
        {
            for(int j = 0;j < S;j++)
            {
                sigma[i][j] = std::powf(k,j)*sigma0*(1<<i);
            }
        }
        cv::Mat samplePyramid[O];
        grayscale.convertTo(samplePyramid[0],CV_32F);
        for(int i = 1;i < O;i++)
        {
            samplePyramid[i] = downsampling(grayscale,1<<i);
            samplePyramid[i].convertTo(samplePyramid[i],CV_32F);
        }
        cv::Mat GuassianPyramid[O][S];
        cv::Mat DoG[O][S-1];
        for(int i = 0;i < O;i++)
        {
            for(int j = 0;j < S;j++)
            {
                int dim = int(6*sigma[i][j] + 1);
                dim |= 1;
                cv::GaussianBlur(samplePyramid[i],GuassianPyramid[i][j],cv::Size(dim,dim),sigma[i][j]);
                if(j > 0)
                {
                    DoG[i][j-1] = GuassianPyramid[i][j] - GuassianPyramid[i][j-1];
                }
            }
        }

        // locate keypoints
        float contrastThreshold = 0.04;
        float edgeThreshold = 10.0;
        int BinNum = 36;
        float SIFT_ORI_SIG_FCTR = 1.52;
        float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;
        float SIFT_ORI_PEAK_RATIO = 0.8;
        float SIFT_INT_DESCR_FCTR = 512.0;
        float SIFT_FIXPT_SCALE = 1;

        keypoints.clear();
        int O_ = O;
        int S_ = S-1;
        for(int o = 0;o < O_;o++)
        {
            for(int s = 1;s < S_-1;s++)
            {
                float threshold = 0.5*contrastThreshold/(n*255*SIFT_FIXPT_SCALE);
                cv::Mat img_prev = DoG[o][s-1], img = DoG[o][s], img_next = DoG[o][s+1];
                for(int i = 0;i < img.rows;i++)
                {
                    for(int j = 0;j < img.cols;j++)
                    {
                        float val = img.at<float>(i,j);
                        int r_start = std::max(0, i - 1), r_end = std::min(i + 2, img.rows), c_start = std::max(0, j - 1), c_end = std::min(j + 2, img.cols);

                        if(std::abs(val) > threshold)
                        {
                            bool is_extreme = true;
                            if(val > 0)
                            {
                                for(int r = r_start;r < r_end;r++)
                                {
                                    for(int c = c_start;c < c_end;c++)
                                    {
                                        if(i == r && j == c)
                                        {
                                            is_extreme = (  (val >= img_prev.at<float>(r,c)) && (val >= img_next.at<float>(r,c))  );
                                        }
                                        else
                                        {
                                            is_extreme = (  (val >= img.at<float>(r,c)) && (val >= img_prev.at<float>(r,c)) && (val >= img_next.at<float>(r,c))  );
                                        }
                                        if(!is_extreme){ break;}
                                    }
                                    if(!is_extreme){ break; }
                                }
                            }
                            else
                            {
                                for(int r = r_start;r < r_end;r++)
                                {
                                    for(int c = c_start;c < c_end;c++)
                                    {
                                        if(i == r && j == c)
                                        {
                                            is_extreme = (  (val <= img_prev.at<float>(r,c)) && (val <= img_next.at<float>(r,c))  );
                                        }
                                        else
                                        {
                                            is_extreme = (  (val <= img.at<float>(r,c)) && (val <= img_prev.at<float>(r,c)) && (val <= img_next.at<float>(r,c))  );
                                        }
                                        if(!is_extreme){ break;}
                                    }
                                    if(!is_extreme){ break; }
                                }
                            }

                            if(is_extreme)
                            {
                                // adjust Local Extrema
                                int SIFT_MAX_INTERP_STEPS = 5;
                                int SIFT_IMG_BORDER = 5;

                                float img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE);
                                float deriv_scale = img_scale * 0.5;
                                float second_deriv_scale = img_scale;
                                float cross_deriv_scale = img_scale * 0.25;

                                cv::Mat img = DoG[o][s];
                                int x = i, y = j, s_ = s;
                                cv::Mat dD;
                                float xi = 0,xr = 0,xc = 0,dxx = 0,dyy = 0,dxy = 0;
                                int ii = 0;
                                bool rejected = false;
                                while(ii < SIFT_MAX_INTERP_STEPS)
                                {
                                    if(s_ < 1 or s_ > n or y < SIFT_IMG_BORDER or y >= img.cols - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= img.rows - SIFT_IMG_BORDER)
                                    {
                                        rejected = true;
                                        break;
                                    }

                                    img = DoG[o][s_];
                                    cv::Mat prev = DoG[o][s_ - 1];
                                    cv::Mat next = DoG[o][s_ + 1];

                                    dD = (cv::Mat_<float >(3,1) <<  ( (img.at<float>(x,y + 1) - img.at<float>(x, y - 1)) * deriv_scale ) ,
                                                                                        ( (img.at<float >(x + 1, y) - img.at<float >(x - 1, y)) * deriv_scale ) ,
                                                                                        ( (next.at<float >(x, y) - prev.at<float >(x, y)) * deriv_scale )  );

                                    float v2 = img.at<float >(x, y) * 2;
                                    dxx = (img.at<float >(x, y + 1) + img.at<float >(x, y - 1) - v2) * second_deriv_scale;
                                    dyy = (img.at<float >(x + 1, y) + img.at<float >(x - 1, y) - v2) * second_deriv_scale;
                                    float dss = (next.at<float >(x, y) + prev.at<float >(x, y) - v2) * second_deriv_scale;
                                    dxy = (img.at<float >(x + 1, y + 1) - img.at<float >(x + 1, y - 1) - img.at<float >(x - 1, y + 1) + img.at<float >(x - 1, y - 1)) * cross_deriv_scale;
                                    float dxs = (next.at<float >(x, y + 1) - next.at<float >(x, y - 1) - prev.at<float >(x, y + 1) + prev.at<float >(x, y - 1)) * cross_deriv_scale;
                                    float dys = (next.at<float >(x + 1, y) - next.at<float >(x - 1, y) - prev.at<float >(x + 1, y) + prev.at<float >(x - 1, y)) * cross_deriv_scale;

                                    cv::Mat H = (cv::Mat_<float >(3,3) << dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss );
                                    cv::Mat H_invert;
                                    cv::invert(H,H_invert);
                                    cv::Mat X = H_invert * dD;


                                    xi = -X.at<float >(2);
                                    xr = -X.at<float >(1);
                                    xc = -X.at<float >(0);

                                    if(std::abs(xi) < 0.5 and std::abs(xr) < 0.5 and std::abs(xc) < 0.5)
                                    {
                                        break;
                                    }
                                    y += int(std::roundf(xc));
                                    x += int(std::roundf(xr));
                                    s_ += int(std::roundf(xi));
                                    ii+=1;
                                }

                                if(!rejected)
                                {
                                    if(ii >= SIFT_MAX_INTERP_STEPS)
                                    {
                                        continue;
                                    }
                                    if(s_ < 1 or s_ > n or y < SIFT_IMG_BORDER or y >= img.cols - SIFT_IMG_BORDER or \
                                    x < SIFT_IMG_BORDER or x >= img.rows - SIFT_IMG_BORDER)
                                    {
                                        continue;
                                    }
                                    float t = dD.at<float >(0)*xc + dD.at<float >(1)*xr + dD.at<float >(2)*xi;
                                    float contr = img.at<float >(x,y) * img_scale + t * 0.5;
                                    if(std::abs(contr) * n < contrastThreshold)
                                    {
                                        continue;
                                    }

                                    float tr = dxx + dyy;
                                    float det = dxx * dyy - dxy * dxy;
                                    if(det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det)
                                    {
                                        continue;
                                    }

                                    Keypoint kpt{};
                                    kpt.o = o;
                                    kpt.s = s_;
                                    kpt.r = (x + xr) * (1 << o);
                                    kpt.c = (y + xc) * (1 << o);
                                    kpt.scale = SIFT_SIGMA * std::pow(2.0, (s_ + xi) / n)*(1 << o) * 2;

                                    float scl_octv = kpt.scale*0.5/(1 << o);
                                    // Get Main Direction
                                    float omax;
                                    std::vector<float> hist;
                                    GetMainDirection(GuassianPyramid[o][s_],x,y,int(std::roundf(SIFT_ORI_RADIUS * scl_octv)),SIFT_ORI_SIG_FCTR * scl_octv,BinNum,omax,hist);
//                                    std::cout << omax << std::endl;
                                    float mag_thr = omax * SIFT_ORI_PEAK_RATIO;
                                    int l = 0,r2 = 0;
                                    for(int k = 0;k < BinNum;k++)
                                    {
                                        if(k > 0)
                                        {
                                            l = k - 1;
                                        }
                                        else
                                        {
                                            l = BinNum - 1;
                                        }
                                        if(k < BinNum - 1)
                                        {
                                            r2 = k + 1;
                                        }
                                        else
                                        {
                                            r2 = 0;
                                        }

                                        if(hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr)
                                        {
                                            int bin = k + 0.5 * (hist[l]-hist[r2]) /(hist[l] - 2 * hist[k] + hist[r2]);
                                            if(bin < 0)
                                            {
                                                bin = BinNum + bin;
                                            }
                                            else
                                            {
                                                if(bin >= BinNum)
                                                {
                                                    bin = bin - BinNum;
                                                }
                                            }
                                            kpt.theta = (360.0/BinNum) * bin;
                                            keypoints.push_back(kpt);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        int SIFT_DESCR_WIDTH = 4;
        int SIFT_DESCR_HIST_BINS = 8;
        int d = SIFT_DESCR_WIDTH;
        n = SIFT_DESCR_HIST_BINS;
        for(int i = 0;i < keypoints.size();i++)
        {
            Keypoint& kpt = keypoints[i];
            float scale = 1.0 / (1 << kpt.o);
            calcSIFTDescriptor(GuassianPyramid[kpt.o][kpt.s], kpt.c * scale, kpt.r * scale, kpt.theta, kpt.scale * scale * 0.5, d, n,kpt.descriptor);
        }
//        for(int i = 0;i < O;i++)
//        {
//            for(int j = 0;j < S-1;j++)
//            {
//                cv::imshow("DoG",DoG[i][j]);
//                cv::waitKey(100);
//            }
//        }
    }

    void match(const Keypoints &src1, const Keypoints &src2, Keypoints &match1, Keypoints &match2, int max_pairs) {
//        std::cout << src1.size() << std::endl;
        double* Data1 = new double[src1.size()*128];
        int i = 0;
        for(auto& kpt:src1)
        {
            memcpy(&Data1[i*128],kpt.descriptor,128* sizeof(double));
            i++;
        }
        cv::Mat _Data1(src1.size(),128,CV_64F, Data1);  //Shallow copy.

//        std::cout << src2.size() << std::endl;
        double* Data2 = new double[src2.size()*128];
        i = 0;
        for(auto& kpt:src2)
        {
            memcpy(&Data2[i*128],kpt.descriptor,128* sizeof(double));
            i++;
        }
        cv::Mat _Data2(src2.size(),128,CV_64F,Data2); //Shallow copy.
        cv::flann::GenericIndex< cvflann::L2<double > > kd(_Data1,cvflann::KDTreeIndexParams(4));  // also Shallow copy of _Data1

        cv::Mat ind(_Data2.rows,1,CV_32S), dis(_Data2.rows,1,CV_64F);
        kd.knnSearch(_Data2,ind,dis,1,cvflann::SearchParams(32));
        int N = std::min(max_pairs,_Data2.rows);
        delete [] Data1;  // the Mat _Data1 should not be used after this, because it's a Shallow copy of the array.
        delete [] Data2; // the Mat _Data2 should not be used after this, because it's a Shallow copy of the array.
        // kd should not be used after this too, because it contains a Shallow copy of _Data1, which is a Shallow copy of array Data1.


        match1.clear(),match2.clear();
        if(N == 0){ return; }
        match1.reserve(N);
        match2.reserve(N);

        std::vector<int> index;
        index.reserve(N);
        std::vector<double> distance;
        distance.reserve(N);
        index.push_back(0);
        distance.push_back(dis.at<double >(0));

        for(i = 1;i < dis.rows;i++)
        {
            double cur_dis = dis.at<double >(i);
            if( index.size() < N || cur_dis < distance[distance.size()-1])
            {
                auto it = std::lower_bound(distance.begin(),distance.end(),cur_dis);
                auto iti = index.begin() + (it - distance.begin());
                distance.insert(it,cur_dis);
                index.insert(iti,i);
                if(index.size() > N)
                {
                    index.pop_back();
                    distance.pop_back();
                }
            }
        }

        for(auto& I:index)
        {
            match2.push_back(src2[I]);
            match1.push_back(src1[ind.at<int>(I)]);
        }

    }

    cv::Mat linkPoints(const cv::Mat &img1, const Keypoints &match1, const cv::Mat &img2, const Keypoints &match2) {
        int Rows = std::max(img1.rows,img2.rows);
        cv::Mat res = cv::Mat::zeros(Rows,img1.cols+img2.cols,CV_8UC3);
        cv::Mat roi1 = res(cv::Rect(0,0,img1.cols,img1.rows));
        cv::Mat roi2 = res(cv::Rect(img1.cols,0,img2.cols,img2.rows));
        img1.copyTo(roi1);
        img2.copyTo(roi2);

        int i = 0;
        for(;i < match1.size();i++)
        {
            cv::line(res,cv::Point(match1[i].c,match1[i].r),cv::Point(match2[i].c + img1.cols,match2[i].r),cv::Scalar(0,0,255));
        }
        return res;
    }
}