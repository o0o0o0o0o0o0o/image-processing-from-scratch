#include <iostream>
#include "Image_Matting.h"

#if defined(_WIN32)
/** This file is part of the Mingw32 package.
     *  unistd.h maps     (roughly) to io.h
     */
    #ifndef _UNISTD_H
        #define _UNISTD_H
        #include <io.h>
        #include <process.h>
    #endif /* _UNISTD_H */
#else
    #include<unistd.h>
#endif


void test_ImageMatting(int N, char* params[])
{
    if(N <= 1)
    {
        std::cout << "usage:" << std::endl;
        std::cout << "./Main image [-s image_scribbles] [-i max_iterations] [-t tolerance]" << std::endl;
    }
    else
    {
        std::string imgname = params[1];
        cv::Mat img;
        img = cv::imread(imgname);
        if(img.empty()){
            std::cout << "image file not found." << std::endl;
            return;
        }

        std::string scribblename;
        cv::Mat img_scribble;
        int max_iters = INT32_MAX;
        double toler = TOL;

        int res;
        int new_N = N-1;
        char* new_params[new_N];
        new_params[0] = params[0];
        for(int i = 2;i < N;i++){new_params[i-1] = params[i];}
        while( (res = getopt(new_N,new_params,"s:i:t:") ) != -1)
        {
            switch(res)
            {
                case 's':
                    scribblename = optarg;
                    break;
                case 'i':
                    max_iters = (int)strtol(optarg, nullptr, 10);
                    break;
                case 't':
                    toler = strtod(optarg, nullptr);
                    break;
                default:
                    std::cout << "usage:" << std::endl;
                    std::cout << "./Main image [-s image_scribbles] [-i max_iterations] [-t tolerance]" << std::endl;
                    return;
                    break;
            }
        }


        if(scribblename.empty())
        {
            img_scribble = ImageMatting::drawScribble(img,imgname.substr(0,imgname.find_last_of('.'))+"_m.png");
        }
        else
        {
            img_scribble = cv::imread(scribblename);
            if(img_scribble.empty()){
                std::cout << "scribble image not found." << std::endl;
                return;
            }
        }

        if(toler <= 0 || toler >= 1)
        {toler=TOL;}
        cv::Mat alpha = ImageMatting::Matting(img, img_scribble, max_iters, toler);
        cv::Mat result = alpha*255;
        result.convertTo(result,CV_8U);
        cv::imwrite("res.png",result);
        cv::imshow("result",result);
        cv::waitKey(0);
    }
}


int main(int argc, char* argv[]) {
    test_ImageMatting(argc,argv);
    return 0;
}