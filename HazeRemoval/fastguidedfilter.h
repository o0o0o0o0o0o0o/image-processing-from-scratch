#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

class FastGuidedFilterImpl;

class FastGuidedFilter
{
public:
    FastGuidedFilter(const cv::Mat &I, int r, double eps,int s);
    ~FastGuidedFilter();

    cv::Mat filter(const cv::Mat &p, int depth = -1) const;

private:
    FastGuidedFilterImpl *impl_;
};

cv::Mat fastGuidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int s = 1,int depth = -1);



static cv::Mat boxfilter(const cv::Mat &I, int r)
{
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r));
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
    if (mat.depth() == depth)
        return mat;

    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

class FastGuidedFilterImpl
{
public:
    FastGuidedFilterImpl(int r, double eps,int s):r(r),eps(eps),s(s){}
    virtual ~FastGuidedFilterImpl() {}

    cv::Mat filter(const cv::Mat &p, int depth);

protected:
    int Idepth,r,s;
    double eps;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class FastGuidedFilterMono : public FastGuidedFilterImpl
{
public:
    FastGuidedFilterMono(const cv::Mat &I, int r, double eps,int s);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:

    cv::Mat I,origI, mean_I, var_I;
};

class FastGuidedFilterColor : public FastGuidedFilterImpl
{
public:
    FastGuidedFilterColor(const cv::Mat &I, int r, double eps,int s);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    std::vector<cv::Mat> origIchannels,Ichannels;
    cv::Mat mean_I_r, mean_I_g, mean_I_b;
    cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};


cv::Mat FastGuidedFilterImpl::filter(const cv::Mat &p, int depth)
{
    cv::Mat p2 = convertTo(p, Idepth);
    cv::resize(p2 ,p2,cv::Size(p2.cols/s,p2.rows/s),0,0);
    cv::Mat result;
    if (p.channels() == 1)
    {
        result = filterSingleChannel(p2);
    }
    else
    {
        std::vector<cv::Mat> pc;
        cv::split(p2, pc);

        for (std::size_t i = 0; i < pc.size(); ++i)
            pc[i] = filterSingleChannel(pc[i]);

        cv::merge(pc, result);
    }

    return convertTo(result, depth == -1 ? p.depth() : depth);
}

FastGuidedFilterMono::FastGuidedFilterMono(const cv::Mat &origI, int r, double eps,int s):FastGuidedFilterImpl(r,eps,s)
{

    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        this->origI = origI.clone();
    else
        this->origI = convertTo(origI, CV_32F);
    cv::resize(this->origI ,I,cv::Size(this->origI.cols/s,this->origI.rows/s),0,0);
    Idepth = I.depth();

    mean_I = boxfilter(I, r);
    cv::Mat mean_II = boxfilter(I.mul(I), r);
    var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat FastGuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{

    cv::Mat mean_p = boxfilter(p, r);
    cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a = boxfilter(a, r);
    cv::Mat mean_b = boxfilter(b, r);
    cv::resize(mean_a ,mean_a,cv::Size(origI.cols,origI.rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_b ,mean_b,cv::Size(origI.cols,origI.rows),0,0,cv::INTER_LINEAR);
    return mean_a.mul(origI) + mean_b;
}

FastGuidedFilterColor::FastGuidedFilterColor(const cv::Mat &origI, int r, double eps, int s):FastGuidedFilterImpl(r,eps,s)// : r(r), eps(eps)
{

    cv::Mat I;
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);
    Idepth = I.depth();

    cv::split(I, origIchannels);
    cv::resize(I,I,cv::Size(I.cols/s,I.rows/s),0,0);
    cv::split(I, Ichannels);

    mean_I_r = boxfilter(Ichannels[0], r);
    mean_I_g = boxfilter(Ichannels[1], r);
    mean_I_b = boxfilter(Ichannels[2], r);

    // variance of I in each local patch: the matrix Sigma.
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), r) - mean_I_r.mul(mean_I_r) + eps;
    cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), r) - mean_I_r.mul(mean_I_g);
    cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), r) - mean_I_r.mul(mean_I_b);
    cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), r) - mean_I_g.mul(mean_I_g) + eps;
    cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), r) - mean_I_g.mul(mean_I_b);
    cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), r) - mean_I_b.mul(mean_I_b) + eps;

    // Inverse of (Sigma + eps * I)
    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

    cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;
}

cv::Mat FastGuidedFilterColor::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r);

    cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
    cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
    cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

    // covariance of (I, p) in each local patch.
    cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

    cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

    cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b);

    cv::Mat mean_a_r = boxfilter(a_r, r);
    cv::Mat mean_a_g = boxfilter(a_g, r);
    cv::Mat mean_a_b = boxfilter(a_b, r);
    cv::Mat mean_b = boxfilter(b, r);
    cv::resize(mean_a_r ,mean_a_r,cv::Size(origIchannels[0].cols,origIchannels[0].rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_a_g ,mean_a_g,cv::Size(origIchannels[1].cols,origIchannels[1].rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_a_b ,mean_a_b,cv::Size(origIchannels[2].cols,origIchannels[2].rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_b,mean_b,cv::Size(origIchannels[2].cols,origIchannels[2].rows),0,0,cv::INTER_LINEAR);
    return (mean_a_r.mul(origIchannels[0]) +mean_a_g.mul(origIchannels[1]) +mean_a_b.mul(origIchannels[2]) + mean_b);

}


FastGuidedFilter::FastGuidedFilter(const cv::Mat &I, int r, double eps,int s)
{
    CV_Assert(I.channels() == 1 || I.channels() == 3);

    if (I.channels() == 1)
        impl_ = new FastGuidedFilterMono(I, 2 * (r/s) + 1, eps,s);
    else
        impl_ = new FastGuidedFilterColor(I, 2 * (r/s) + 1, eps,s);
}

FastGuidedFilter::~FastGuidedFilter()
{
    delete impl_;
}

cv::Mat FastGuidedFilter::filter(const cv::Mat &p, int depth) const
{
    return impl_->filter(p, depth);
}

cv::Mat fastGuidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int s,int depth)
{
    return FastGuidedFilter(I, r, eps,s).filter(p, depth);
}

#endif