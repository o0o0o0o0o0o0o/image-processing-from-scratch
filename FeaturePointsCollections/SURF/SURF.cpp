//
// author: 会飞的吴克
//

#include "SURF.h"

namespace SURF{
    struct SurfHF
    {
        int p0, p1, p2, p3;
        float w;

        SurfHF(): p0(0), p1(0), p2(0), p3(0), w(0) {}
    };

    static void
    resizeHaarPattern( const int src[][5], SurfHF* dst, int n, int oldSize, int newSize, int widthStep )
    {
        float ratio = (float)newSize/oldSize;
        for( int k = 0; k < n; k++ )
        {
            int dx1 = cvRound( ratio*src[k][0] );
            int dy1 = cvRound( ratio*src[k][1] );
            int dx2 = cvRound( ratio*src[k][2] );
            int dy2 = cvRound( ratio*src[k][3] );

            dst[k].p0 = dy1*widthStep + dx1;
            dst[k].p1 = dy2*widthStep + dx1;
            dst[k].p2 = dy1*widthStep + dx2;
            dst[k].p3 = dy2*widthStep + dx2;

            dst[k].w = src[k][4]/((float)(dx2-dx1)*(dy2-dy1));
        }
    }

    inline float calcHaarPattern( const int* origin, const SurfHF* f, int n )
    {
        double d = 0;
        for( int k = 0; k < n; k++ )
            d += (origin[f[k].p0] + origin[f[k].p3] - origin[f[k].p1] - origin[f[k].p2])*f[k].w;
        return (float)d;
    }

    static void calcLayerDetAndTrace( const cv::Mat& sum, int size, int sampleStep, cv::Mat& det, cv::Mat& trace )
    {
        const int NX=3, NY=3, NXY=4; // number of non zero regions in Dxx, Dyy and Dxy.
        const int dx_s[NX][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
        const int dy_s[NY][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
        const int dxy_s[NXY][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} }; //position and values of non-zero regions.

        SurfHF Dx[NX], Dy[NY], Dxy[NXY];
        if( size > sum.rows-1 || size > sum.cols-1 )
            return;

        // resize the patterns.
        resizeHaarPattern( dx_s , Dx , NX , 9, size, sum.cols );
        resizeHaarPattern( dy_s , Dy , NY , 9, size, sum.cols );
        resizeHaarPattern( dxy_s, Dxy, NXY, 9, size, sum.cols );

        /* The integral image 'sum' is one pixel bigger than the source image */
        int samples_i = 1+(sum.rows-1-size)/sampleStep;
        int samples_j = 1+(sum.cols-1-size)/sampleStep;

        /* Ignore pixels where some of the kernel is outside the image */
        int margin = (size/2)/sampleStep;
        for( int i = 0; i < samples_i; i++ )
        {
            const int* sum_ptr = sum.ptr<int>(i*sampleStep);
            float* det_ptr = &det.at<float>(i+margin, margin);
            float* trace_ptr = &trace.at<float>(i+margin, margin);
            for( int j = 0; j < samples_j; j++ )
            {
                // box filtering using integral image.
                float dx  = calcHaarPattern( sum_ptr, Dx , 3 );
                float dy  = calcHaarPattern( sum_ptr, Dy , 3 );
                float dxy = calcHaarPattern( sum_ptr, Dxy, 4 );
                sum_ptr += sampleStep;
                det_ptr[j] = dx*dy - 0.81f*dxy*dxy;
                trace_ptr[j] = dx + dy;
            }
        }
    }

    static bool
    interpolateKeypoint( float N9[3][9], int dx, int dy, int ds, Keypoint& kpt )
    {
        cv::Vec3f b(-(N9[1][5]-N9[1][3])/2,  // Negative 1st deriv with respect to x
                -(N9[1][7]-N9[1][1])/2,  // Negative 1st deriv with respect to y
                -(N9[2][4]-N9[0][4])/2); // Negative 1st deriv with respect to s

        cv::Matx33f A(
                N9[1][3]-2*N9[1][4]+N9[1][5],            // 2nd deriv x, x
                (N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4, // 2nd deriv x, y
                (N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4, // 2nd deriv x, s
                (N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4, // 2nd deriv x, y
                N9[1][1]-2*N9[1][4]+N9[1][7],            // 2nd deriv y, y
                (N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4, // 2nd deriv y, s
                (N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4, // 2nd deriv x, s
                (N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4, // 2nd deriv y, s
                N9[0][4]-2*N9[1][4]+N9[2][4]);           // 2nd deriv s, s

        cv::Vec3f x = A.solve(b, cv::DECOMP_LU);

        bool ok = (x[0] != 0 || x[1] != 0 || x[2] != 0) &&
                  std::abs(x[0]) <= 1 && std::abs(x[1]) <= 1 && std::abs(x[2]) <= 1;

        if( ok )
        {
            kpt.x += x[0]*dx;
            kpt.y += x[1]*dy;
            kpt.size = (float)cvRound( kpt.size + x[2]*ds );
        }
        return ok;
    }

    void findMaximaInLayer( const cv::Mat& sum, const std::vector<cv::Mat>& dets, const std::vector<cv::Mat>& traces,
                                             const std::vector<int>& sizes, Keypoints& keypoints,
                                             int octave, int layer, float hessianThreshold, int sampleStep )
    {
        int size = sizes[layer];

        int layer_rows = (sum.rows-1)/sampleStep;
        int layer_cols = (sum.cols-1)/sampleStep;

        // Ignore pixels without a 3x3x3 neighbourhood in the layer above
        int margin = (sizes[layer+1]/2)/sampleStep+1;
        int step = (int)(dets[layer].step/dets[layer].elemSize());
        for( int i = margin; i < layer_rows - margin; i++ )
        {
            const float* det_ptr = dets[layer].ptr<float>(i);
            const float* trace_ptr = traces[layer].ptr<float>(i);
            for( int j = margin; j < layer_cols-margin; j++ )
            {
                float val0 = det_ptr[j];
                if( val0 > hessianThreshold )
                {
                    /* Coordinates for the start of the wavelet in the sum image. There
                       is some integer division involved, so don't try to simplify this
                       (cancel out sampleStep) without checking the result is the same */
                    int sum_i = sampleStep*(i-(size/2)/sampleStep);
                    int sum_j = sampleStep*(j-(size/2)/sampleStep);

                    /* The 3x3x3 neighbouring samples around the maxima.
                       The maxima is included at N9[1][4] */
                    const float *det1 = &dets[layer-1].at<float>(i, j);
                    const float *det2 = &dets[layer].at<float>(i, j);
                    const float *det3 = &dets[layer+1].at<float>(i, j);

                    float N9[3][9] = { { det1[-step-1], det1[-step], det1[-step+1],
                                               det1[-1]  , det1[0] , det1[1],
                                               det1[step-1] , det1[step] , det1[step+1]  },
                                       { det2[-step-1], det2[-step], det2[-step+1],
                                               det2[-1]  , det2[0] , det2[1],
                                               det2[step-1] , det2[step] , det2[step+1]  },
                                       { det3[-step-1], det3[-step], det3[-step+1],
                                               det3[-1]  , det3[0] , det3[1],
                                               det3[step-1] , det3[step] , det3[step+1]  } };


                    /* Non-maxima suppression. val0 is at N9[1][4]*/
                    if( val0 > N9[0][0] && val0 > N9[0][1] && val0 > N9[0][2] &&
                        val0 > N9[0][3] && val0 > N9[0][4] && val0 > N9[0][5] &&
                        val0 > N9[0][6] && val0 > N9[0][7] && val0 > N9[0][8] &&
                        val0 > N9[1][0] && val0 > N9[1][1] && val0 > N9[1][2] &&
                        val0 > N9[1][3]                    && val0 > N9[1][5] &&
                        val0 > N9[1][6] && val0 > N9[1][7] && val0 > N9[1][8] &&
                        val0 > N9[2][0] && val0 > N9[2][1] && val0 > N9[2][2] &&
                        val0 > N9[2][3] && val0 > N9[2][4] && val0 > N9[2][5] &&
                        val0 > N9[2][6] && val0 > N9[2][7] && val0 > N9[2][8] )
                    {
                        /* Calculate the wavelet center coordinates for the maxima */
                        float center_i = sum_i + (size-1)*0.5f;
                        float center_j = sum_j + (size-1)*0.5f;
                        Keypoint kpt( center_j, center_i, (float)sizes[layer],
                                      -1, val0, octave, CV_SIGN(trace_ptr[j]) );

                        /* Interpolate maxima location within the 3x3x3 neighbourhood  */
                        int ds = size - sizes[layer-1];
                        bool interp_ok = interpolateKeypoint( N9, sampleStep, sampleStep, ds, kpt );

                        /* Sometimes the interpolation step gives a negative size etc. */
                        if( interp_ok )
                        {
                            keypoints.push_back(kpt);
                        }
                    }
                }
            }
        }
    }

    struct SURFInvoker
    {
        enum { ORI_RADIUS = 6, ORI_WIN = 60, PATCH_SZ = 20 };
        SURFInvoker( const cv::Mat& _img, const cv::Mat& _sum,
                     Keypoints& _keypoints, bool _extended )
        {
            keypoints = &_keypoints;
            img = &_img;
            sum = &_sum;
            extended = _extended;

            // Simple bound for number of grid points in circle of radius ORI_RADIUS
            const int nOriSampleBound = (2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);

            // Allocate arrays
            apt.resize(nOriSampleBound);
            aptw.resize(nOriSampleBound);
            DW.resize(PATCH_SZ*PATCH_SZ);

            /* Coordinates and weights of samples used to calculate orientation */
            float SURF_ORI_SIGMA = 2.5f;
            cv::Mat G_ori = cv::getGaussianKernel( 2*ORI_RADIUS+1, SURF_ORI_SIGMA, CV_32F );
            nOriSamples = 0;
            for( int i = -ORI_RADIUS; i <= ORI_RADIUS; i++ )
            {
                for( int j = -ORI_RADIUS; j <= ORI_RADIUS; j++ )
                {
                    if( i*i + j*j <= ORI_RADIUS*ORI_RADIUS )
                    {
                        apt[nOriSamples] = cv::Point(i,j);
                        aptw[nOriSamples++] = G_ori.at<float>(i+ORI_RADIUS,0) * G_ori.at<float>(j+ORI_RADIUS,0);
                    }
                }
            }

            /* Gaussian used to weight descriptor samples */
            float SURF_DESC_SIGMA = 3.3f;
            cv::Mat G_desc = cv::getGaussianKernel( PATCH_SZ, SURF_DESC_SIGMA, CV_32F );
            for( int i = 0; i < PATCH_SZ; i++ )
            {
                for( int j = 0; j < PATCH_SZ; j++ )
                    DW[i*PATCH_SZ+j] = G_desc.at<float>(i,0) * G_desc.at<float>(j,0);
            }
        }

        void operator()(int N) const
        {
            /* X and Y gradient wavelet data */
            const int NX=2, NY=2;
            const int dx_s[NX][5] = {{0, 0, 2, 4, -1}, {2, 0, 4, 4, 1}};
            const int dy_s[NY][5] = {{0, 0, 4, 2, 1}, {0, 2, 4, 4, -1}};

            // Optimisation is better using nOriSampleBound than nOriSamples for
            // array lengths.  Maybe because it is a constant known at compile time
            const int nOriSampleBound =(2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);

            float X[nOriSampleBound], Y[nOriSampleBound], angle[nOriSampleBound];
            uchar PATCH[PATCH_SZ+1][PATCH_SZ+1];
            float DX[PATCH_SZ][PATCH_SZ], DY[PATCH_SZ][PATCH_SZ];
            cv::Mat _patch(PATCH_SZ+1, PATCH_SZ+1, CV_8U, PATCH);

            int dsize = extended ? 128 : 64;

            int k, k1 = 0, k2 = N;
            double maxSize = 0;
            for( k = k1; k < k2; k++ )
            {
                maxSize = std::max(maxSize, (*keypoints)[k].size);
            }
            int imaxSize = std::max(cvCeil((PATCH_SZ+1)*maxSize*1.2f/9.0f), 1);
            cv::AutoBuffer<uchar> winbuf(imaxSize*imaxSize);

            for( k = k1; k < k2; k++ )
            {
                int i, j, kk, nangle;
                double* vec;
                SurfHF dx_t[NX], dy_t[NY];
                Keypoint& kp = (*keypoints)[k];
                float size = kp.size;
                cv::Point2f center{static_cast<float>(kp.x),static_cast<float>(kp.y)};
                /* The sampling intervals and wavelet sized for selecting an orientation
                 and building the keypoint descriptor are defined relative to 's' */
                float s = size*1.2f/9.0f;
                /* To find the dominant orientation, the gradients in x and y are
                 sampled in a circle of radius 6s using wavelets of size 4s.
                 We ensure the gradient wavelet size is even to ensure the
                 wavelet pattern is balanced and symmetric around its center */
                int grad_wav_size = 2*cvRound( 2*s );
                if( sum->rows < grad_wav_size || sum->cols < grad_wav_size )
                {
                    /* when grad_wav_size is too big,
                     * the sampling of gradient will be meaningless
                     * mark keypoint for deletion. */
                    kp.size = -1;
                    continue;
                }

                float descriptor_dir = 360.f - 90.f;

                resizeHaarPattern( dx_s, dx_t, NX, 4, grad_wav_size, sum->cols );
                resizeHaarPattern( dy_s, dy_t, NY, 4, grad_wav_size, sum->cols );
                for( kk = 0, nangle = 0; kk < nOriSamples; kk++ )
                {
                    int x = cvRound( center.x + apt[kk].x*s - (float)(grad_wav_size-1)/2 );
                    int y = cvRound( center.y + apt[kk].y*s - (float)(grad_wav_size-1)/2 );
                    if( y < 0 || y >= sum->rows - grad_wav_size ||
                        x < 0 || x >= sum->cols - grad_wav_size )
                        continue;
                    const int* ptr = &sum->at<int>(y, x);
                    float vx = calcHaarPattern( ptr, dx_t, 2 );
                    float vy = calcHaarPattern( ptr, dy_t, 2 );
                    X[nangle] = vx*aptw[kk];
                    Y[nangle] = vy*aptw[kk];
                    nangle++;
                }
                if( nangle == 0 )
                {
                    // No gradient could be sampled because the keypoint is too
                    // near too one or more of the sides of the image. As we
                    // therefore cannot find a dominant direction, we skip this
                    // keypoint and mark it for later deletion from the sequence.
                    kp.size = -1;
                    continue;
                }

                cv::phase( cv::Mat(1, nangle, CV_32F, X), cv::Mat(1, nangle, CV_32F, Y), cv::Mat(1, nangle, CV_32F, angle), true );

                float bestx = 0, besty = 0, descriptor_mod = 0;
                int SURF_ORI_SEARCH_INC = 5;
                for( i = 0; i < 360; i += SURF_ORI_SEARCH_INC )
                {
                    float sumx = 0, sumy = 0, temp_mod;
                    for( j = 0; j < nangle; j++ )
                    {
                        int d = std::abs(cvRound(angle[j]) - i);
                        if( d < ORI_WIN/2 || d > 360-ORI_WIN/2 )
                        {
                            sumx += X[j];
                            sumy += Y[j];
                        }
                    }
                    temp_mod = sumx*sumx + sumy*sumy;
                    if( temp_mod > descriptor_mod )
                    {
                        descriptor_mod = temp_mod;
                        bestx = sumx;
                        besty = sumy;
                    }
                }
                descriptor_dir = cv::fastAtan2( -besty, bestx );
                kp.theta = descriptor_dir;

                /* Extract a window of pixels around the keypoint of size 20s */
                int win_size = (int)((PATCH_SZ+1)*s);
                cv::Mat win(win_size, win_size, CV_8U, winbuf.data());

                descriptor_dir *= (float)(CV_PI/180);
                float sin_dir = -std::sin(descriptor_dir);
                float cos_dir =  std::cos(descriptor_dir);


                float win_offset = -(float)(win_size-1)/2;
                float start_x = center.x + win_offset*cos_dir + win_offset*sin_dir;
                float start_y = center.y - win_offset*sin_dir + win_offset*cos_dir;
                uchar* WIN = win.data;

                int ncols1 = img->cols-1, nrows1 = img->rows-1;
                size_t imgstep = img->step;
                for( i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir )
                {
                    double pixel_x = start_x;
                    double pixel_y = start_y;
                    for( j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir )
                    {
                        int ix = cvFloor(pixel_x), iy = cvFloor(pixel_y);
                        if( (unsigned)ix < (unsigned)ncols1 &&
                            (unsigned)iy < (unsigned)nrows1 )
                        {
                            float a = (float)(pixel_x - ix), b = (float)(pixel_y - iy);
                            const uchar* imgptr = &img->at<uchar>(iy, ix);
                            WIN[i*win_size + j] = (uchar)
                                    cvRound(imgptr[0]*(1.f - a)*(1.f - b) +
                                            imgptr[1]*a*(1.f - b) +
                                            imgptr[imgstep]*(1.f - a)*b +
                                            imgptr[imgstep+1]*a*b);
                        }
                        else
                        {
                            int x = std::min(std::max(cvRound(pixel_x), 0), ncols1);
                            int y = std::min(std::max(cvRound(pixel_y), 0), nrows1);
                            WIN[i*win_size + j] = img->at<uchar>(y, x);
                        }
                    }
                }

                // Scale the window to size PATCH_SZ so each pixel's size is s. This
                // makes calculating the gradients with wavelets of size 2s easy
                resize(win, _patch, _patch.size(), 0, 0, cv::INTER_AREA);

                // Calculate gradients in x and y with wavelets of size 2s
                for( i = 0; i < PATCH_SZ; i++ )
                    for( j = 0; j < PATCH_SZ; j++ )
                    {
                        float dw = DW[i*PATCH_SZ + j];
                        float vx = (PATCH[i][j+1] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i+1][j])*dw;
                        float vy = (PATCH[i+1][j] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i][j+1])*dw;
                        DX[i][j] = vx;
                        DY[i][j] = vy;
                    }

                // Construct the descriptor
                vec = kp.descriptor;
                for( kk = 0; kk < dsize; kk++ )
                    vec[kk] = 0;
                double square_mag = 0;
                if( extended )
                {
                    // 128-bin descriptor
                    for( i = 0; i < 4; i++ )
                        for( j = 0; j < 4; j++ )
                        {
                            for(int y = i*5; y < i*5+5; y++ )
                            {
                                for(int x = j*5; x < j*5+5; x++ )
                                {
                                    float tx = DX[y][x], ty = DY[y][x];
                                    if( ty >= 0 )
                                    {
                                        vec[0] += tx;
                                        vec[1] += (float)fabs(tx);
                                    } else {
                                        vec[2] += tx;
                                        vec[3] += (float)fabs(tx);
                                    }
                                    if ( tx >= 0 )
                                    {
                                        vec[4] += ty;
                                        vec[5] += (float)fabs(ty);
                                    } else {
                                        vec[6] += ty;
                                        vec[7] += (float)fabs(ty);
                                    }
                                }
                            }
                            for( kk = 0; kk < 8; kk++ )
                                square_mag += vec[kk]*vec[kk];
                            vec += 8;
                        }
                }
                else
                {
                    // 64-bin descriptor
                    for( i = 0; i < 4; i++ )
                        for( j = 0; j < 4; j++ )
                        {
                            for(int y = i*5; y < i*5+5; y++ )
                            {
                                for(int x = j*5; x < j*5+5; x++ )
                                {
                                    float tx = DX[y][x], ty = DY[y][x];
                                    vec[0] += tx; vec[1] += ty;
                                    vec[2] += (float)fabs(tx); vec[3] += (float)fabs(ty);
                                }
                            }
                            for( kk = 0; kk < 4; kk++ )
                                square_mag += vec[kk]*vec[kk];
                            vec+=4;
                        }
                }

                // unit vector is essential for contrast invariance
                vec = kp.descriptor;
                float scale = (float)(1./(std::sqrt(square_mag) + FLT_EPSILON));
                for( kk = 0; kk < dsize; kk++ )
                    vec[kk] *= scale;
            }
        }

        const cv::Mat* img;
        const cv::Mat* sum;
        Keypoints* keypoints;
        bool extended;

        int nOriSamples;
        std::vector<cv::Point> apt;
        std::vector<float> aptw;
        std::vector<float> DW;
    };

    void extract(const cv::Mat &img, Keypoints& keypoints) {
        cv::Mat gray,integral;
        cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
        cv::integral(gray,integral,CV_32S);
        // the integrals are saved as integers, so if the image gets too big
        // then this algorithm may fail due to integer overflow.
        // Also it might still work as well despite that overflow occurs.
        // because hopefully the difference of the integrals hold the same even if there're overflows.


        const int hessianThreshold = 100;
        const int nOctaves = 4;
        const int nOctaveLayers = 3; // inner layers.

        const int SAMPLE_STEP0 = 1;
        int nTotalLayers = (nOctaveLayers+2)*nOctaves; // total layers.
        int nMiddleLayers = nOctaveLayers*nOctaves; // total inner layers.

        std::vector<cv::Mat> dets(nTotalLayers);
        std::vector<cv::Mat> traces(nTotalLayers);
        std::vector<int> sizes(nTotalLayers);
        std::vector<int> sampleSteps(nTotalLayers);
        std::vector<int> middleIndices(nMiddleLayers);

        // Allocate space and calculate properties of each layer
        int SURF_HAAR_SIZE0 = 9, SURF_HAAR_SIZE_INC = 6;
        int index = 0, middleIndex = 0, step = SAMPLE_STEP0;
        for( int octave = 0; octave < nOctaves; octave++ )
        {
            for( int layer = 0; layer < nOctaveLayers+2; layer++ )
            {
                /* The integral image sum is one pixel bigger than the source image*/
                dets[index].create( (integral.rows-1)/step, (integral.cols-1)/step, CV_32F );
                traces[index].create( (integral.rows-1)/step, (integral.cols-1)/step, CV_32F );
                sizes[index] = (SURF_HAAR_SIZE0 + SURF_HAAR_SIZE_INC*layer) << octave;
                sampleSteps[index] = step;

                if( 0 < layer && layer <= nOctaveLayers )
                    middleIndices[middleIndex++] = index;
                index++;
            }
            step *= 2;
        }

        for(int i = 0;i < nTotalLayers;i++)
        {
            calcLayerDetAndTrace( integral, sizes[i], sampleSteps[i], dets[i], traces[i] );
        }

        keypoints.clear();
        for(int i = 0;i < nMiddleLayers;i++)
        {
            int layer = middleIndices[i];
            int octave = i / nOctaveLayers;
            findMaximaInLayer( integral, dets, traces, sizes,
                               keypoints, octave, layer, hessianThreshold,
                               sampleSteps[layer] );
        }


        auto cmp = [](const Keypoint& kp1, const Keypoint& kp2){
            {
                if(kp1.response > kp2.response) return true;
                if(kp1.response < kp2.response) return false;
                if(kp1.size > kp2.size) return true;
                if(kp1.size < kp2.size) return false;
                if(kp1.o > kp2.o) return true;
                if(kp1.o < kp2.o) return false;
                if(kp1.y > kp2.y) return true;
                if(kp1.y < kp2.y) return false;
                return kp1.x < kp2.x;
            }
        };
        std::sort(keypoints.begin(), keypoints.end(), cmp);

        int i, j, N = (int)keypoints.size();
        if( N > 0 )
        {
            SURFInvoker(gray, integral, keypoints, true)(N);
            for( i = j = 0; i < N; i++ )
            {
                if( keypoints[i].size > 0 )
                {
                    if( i > j )
                    {
                        keypoints[j] = keypoints[i];
                    }
                    j++;
                }
            }
            if( N > j )
            {
                N = j;
                keypoints.resize(N);
            }
        }

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
            cv::line(res,cv::Point(match1[i].x,match1[i].y),cv::Point(match2[i].x + img1.cols,match2[i].y),cv::Scalar(0,0,255));
        }
        return res;
    }
}


// I modified, reorganized and reused some of the codes in opencv for this part, so
// I put the original copyright notice and disclaimer here.
/////////////////////////////////////////////////////////////////////////////////
//* Copyright© 2008, Liu Liu All rights reserved.
//*
//* Redistribution and use in source and binary forms, with or
//* without modification, are permitted provided that the following
//        * conditions are met:
//*  Redistributions of source code must retain the above
//*  copyright notice, this list of conditions and the following
//        *  disclaimer.
//*  Redistributions in binary form must reproduce the above
//*  copyright notice, this list of conditions and the following
//        *  disclaimer in the documentation and/or other materials
//*  provided with the distribution.
//*  The name of Contributor may not be used to endorse or
//*  promote products derived from this software without
//*  specific prior written permission.
//*
//* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
//        * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//        * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//* DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
//* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//        * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
//* OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//        * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
//        * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//        * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
//* OF SUCH DAMAGE.
//*/
//////////////////////////////////////////////////////////////////////////////////
