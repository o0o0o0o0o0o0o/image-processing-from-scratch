//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_WATERSHED_H
#define IMAGEPROCESSINGFROMSCRATCH_WATERSHED_H
#include <iostream>
#include <opencv2/opencv.hpp>

namespace Segmentation{

    const int SHED = -1;
    const int INQUEUE = -2;

    struct point2D{
        int r,c;
    };
    struct RGB{
        uchar r,g,b;
    };

    cv::Mat WaterShed_(const cv::Mat & gray_img, const cv::Mat & markers)
    /*
     * gray_img: a 1 channel 8-bit img.
     * markers: different basins marked as different positive integers,
     *         other position is 0.(32-bit 1 channel img)
     *
     * return: watershed mask.(1 channel)
     */
    {
        assert(!gray_img.empty());
        assert(gray_img.channels() == 1 && markers.channels() == 1);
        assert(gray_img.rows == markers.rows && gray_img.cols == markers.cols);
        const int graylevelnum = 256;

        cv::Mat gray;
        gray_img.convertTo(gray,CV_8U);

        cv::Mat res;
        markers.convertTo(res,CV_32S);

        std::vector<std::queue<point2D> > queues;
        queues.reserve(graylevelnum);
        for(int i = 0;i < graylevelnum;i++)
        {
            queues.emplace_back(std::queue<point2D>());
        }

        int Rows = res.rows - 1, Cols = res.cols - 1;
        int *p = res.ptr<int>(0);
        for(int i = 0;i < res.cols;i++){p[i] = SHED;}
        p = res.ptr<int>(Rows);
        for(int i = 0;i < res.cols;i++){p[i] = SHED;}

        for(int i = 1;i < Rows;i++)
        {
            int *res_p_pre = res.ptr<int >(i-1);
            int *res_p_cur = res.ptr<int >(i);
            int *res_p_next = res.ptr<int >(i+1);

            uchar *gray_p = gray.ptr<uchar >(i);
            res_p_cur[0] = SHED; res_p_cur[Cols] = SHED;
            for(int j = 1;j < Cols;j++)
            {
                if(res_p_cur[j] < 0){res_p_cur[j] = 0;}

                //check 4 neighbors.
                int intensity = graylevelnum;
                if(res_p_cur[j] == 0 && (res_p_pre[j] > 0 || res_p_next[j] > 0 || res_p_cur[j-1] > 0 || res_p_cur[j+1] > 0)){intensity = gray_p[j];}
                if(intensity < graylevelnum){queues[intensity].push({i,j});res_p_cur[j] = INQUEUE;}
            }
        }

        int ind = 0;
        for(;ind < graylevelnum;ind++)
        {
            if(!queues[ind].empty()){ break;}
        }

        if(ind < graylevelnum)
        {
            int lab, t;
            while(true)
            {
                if(queues[ind].empty())
                {
                    for(;ind < graylevelnum;ind++)
                    {
                        if(!queues[ind].empty()){ break;}
                    }
                }
                if(ind >= graylevelnum){ break;}

                point2D cur = queues[ind].front();
                queues[ind].pop();

                //std::cout << ind << " " << cur.r << " " << cur.c << std::endl;

                //mark current pixel.
                lab = 0;
                t = res.at<int>(cur.r-1,cur.c);
                if(t > 0) lab = t;
                t = res.at<int>(cur.r+1,cur.c);
                if(t > 0)
                {
                    if(lab == 0){lab = t;}
                    else if(lab != t){lab = SHED;}
                }
                t = res.at<int>(cur.r,cur.c-1);
                if(t > 0)
                {
                    if(lab == 0){lab = t;}
                    else if(lab != t){lab = SHED;}
                }
                t = res.at<int>(cur.r,cur.c+1);
                if(t > 0)
                {
                    if(lab == 0){lab = t;}
                    else if(lab != t){lab = SHED;}
                }

                assert(lab != 0);
                res.at<int>(cur.r,cur.c) = lab;
                //std::cout << "lab:" << lab << std::endl;

                if(lab != SHED)
                {
                    //check 4 neighbors for unmarked pixels.
                    int r_ind = cur.r-1, c_ind = cur.c;
                    if(res.at<int>(r_ind,c_ind) == 0)
                    {
                        t = gray.at<uchar >(r_ind,c_ind);
                        //std::cout << "t:" << t << std::endl;
                        queues[t].push({r_ind,c_ind});
                        res.at<int>(r_ind,c_ind) = INQUEUE;
                        ind = ind < t ? ind : t;
                    }
                    r_ind = cur.r+1; c_ind = cur.c;
                    if(res.at<int>(r_ind,c_ind) == 0)
                    {
                        t = gray.at<uchar >(r_ind,c_ind);
                        //std::cout << "t:" << t << std::endl;
                        queues[t].push({r_ind,c_ind});
                        res.at<int>(r_ind,c_ind) = INQUEUE;
                        ind = ind < t ? ind : t;
                    }
                    r_ind = cur.r; c_ind = cur.c-1;
                    if(res.at<int>(r_ind,c_ind) == 0)
                    {
                        t = gray.at<uchar >(r_ind,c_ind);
                        //std::cout << "t:" << t << std::endl;
                        queues[t].push({r_ind,c_ind});
                        res.at<int>(r_ind,c_ind) = INQUEUE;
                        ind = ind < t ? ind : t;
                    }
                    r_ind = cur.r; c_ind = cur.c+1;
                    if(res.at<int>(r_ind,c_ind) == 0)
                    {
                        t = gray.at<uchar >(r_ind,c_ind);
                        //std::cout << "t:" << t << std::endl;
                        queues[t].push({r_ind,c_ind});
                        res.at<int>(r_ind,c_ind) = INQUEUE;
                        ind = ind < t ? ind : t;
                    }
                }
            }
        }

        return res;
    }

    cv::Mat ColorMask(const cv::Mat & mask, const int color_nums)
    /*
     * mask: watershed mask(1 channel) with color_nums basins.
     * color_nums: number of different areas.(except SHED)
     *
     * return: segmented colored image(3 channel)
     */
    {
        assert(mask.channels() == 1 && color_nums > 0);
        cv::Mat Mask;
        mask.convertTo(Mask,CV_32S);
        cv::Mat res = cv::Mat::zeros(mask.rows,mask.cols,CV_8UC3);

        //generate colors
        const int graylevelnum = 256;
        std::vector<RGB> colors;
        colors.reserve(color_nums);
        double upper_bound = double(graylevelnum - 1);
        double step =  upper_bound / ceil(pow(color_nums,1/3.0));
        double semi_step = step/2;
        double semi_semi_step = semi_step / 2;

        //std::cout << step << std::endl;
        srand(256);
        int i = 0;
        for(double r = semi_step;r < upper_bound;r+=step)
        {
            for(double g = semi_step;g < upper_bound;g+=step)
            {
                for(double b = semi_step;b < upper_bound;b+=step)
                {
                    uchar R = r + (rand()*semi_step/RAND_MAX - semi_semi_step);
                    uchar G = g + (rand()*semi_step/RAND_MAX - semi_semi_step);
                    uchar B = b + (rand()*semi_step/RAND_MAX - semi_semi_step);
                    //std::cout << int(R) <<" "<< int(G) <<" "<< int(B) << std::endl;
                    //std::cout << int(uchar(r)) <<" "<< int(uchar(g)) <<" "<< int(uchar(b)) << std::endl;
                    colors.push_back({R,G,B});
                    i++;
                    if(i == color_nums){break;}
                }
                if(i == color_nums){break;}
            }
            if(i == color_nums){break;}
        }
        //std::cout << colors.size() << std::endl;

        for(i = 1;i < res.rows - 1;i++)
        {
            cv::Vec3b *p =  res.ptr<cv::Vec3b >(i);
            int *Mp = Mask.ptr<int >(i);
            for(int j = 1;j < res.cols - 1;j++)
            {
                if(Mp[j] > 0)
                {
                    RGB cur = colors[Mp[j]-1];
                    p[j][0] = cur.b;
                    p[j][1] = cur.g;
                    p[j][2] = cur.r;
                    //std::cout << Mp[j] << " " << int(cur.r) <<" "<< int(cur.g) <<" "<< int(cur.b) << std::endl;
                }
            }
        }

        return res;
    }

    ////////////with user specified markers/////////////////

    void onMouse(int event, int x, int y, int flag, void *param)
    {
        std::pair<cv::Point,bool> &P = *(std::pair<cv::Point,bool> *)param;
        switch (event)
        {
            //when mouse move
            case cv::EVENT_MOUSEMOVE:
            {
                P.first = cv::Point(x, y);
            }
                break;
            //on LBUTTONDOWN
            case cv::EVENT_LBUTTONDOWN:
            {
                P.second = true;
                P.first = cv::Point(x, y);
            }
                break;
            //on LBUTTONUP
            case cv::EVENT_LBUTTONUP:
            {
                P.second = false;
            }
                break;
            default:
                break;
        }
    }

    cv::Mat WaterShedWithMarkers(const cv::Mat & img) //let user draw markers interactively.
    /*
     * img: source image.
     *
     * return: watershed result.
     */
    {
        cv::Mat Img = img.clone();
        cv::Mat markers = cv::Mat::zeros(Img.rows,Img.cols,CV_32S);

        cv::Point CurPoint;
        bool Drawing = false;
        auto param = std::make_pair(CurPoint,Drawing);

        cv::namedWindow("Mouse Event", 0);
        cv::setMouseCallback("Mouse Event", onMouse, (void *)&param);
        int Thickness = 0;
        uchar B = 0, G = 0, R = 255;

        cv::namedWindow("slide bar", 0);
        cv::createTrackbar("thickness", "slide bar", &Thickness, 100, nullptr);

        int basins = 0;
        int key;
        bool lastDrawing = false;
        while (true)
        {
            CurPoint = param.first;
            Drawing = param.second;
            if (Drawing)
            {
                circle(Img, CurPoint, 0, cv::Scalar(B, G, R), Thickness + 1);
                circle(markers, CurPoint, 0, cv::Scalar(basins+1), Thickness + 1);
            }
            else
            {
                if(lastDrawing)
                {
                    basins++;
                }
            }

            imshow("Mouse Event", Img);
            key = cv::waitKey(1);
            if (key == 27)
            {
                break;
            }
            if (key == 'c')
            {
                std::cout << "clear" << std::endl;
                Img = img.clone();
                markers = cv::Mat::zeros(Img.rows,Img.cols,CV_32S);
                basins = 0;
            }
            lastDrawing = Drawing;
        }

        cv::Mat gray;
        cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);

        cv::Mat gray_img;
        //cv::bilateralFilter(gray,gray_img,0,10,5); //bilateral filter
        cv::GaussianBlur(gray,gray_img,{3,3},1);
        cv::Mat g_x,g_y;
        cv::Sobel(gray_img,g_x,CV_64F,1,0);
        cv::Sobel(gray_img,g_y,CV_64F,0,1);
        cv::Mat grad = g_x.mul(g_x) + g_y.mul(g_y);
        cv::sqrt(grad,grad);
        double minv = 0.0, maxv = 0.0;
        double* minp = &minv;
        double* maxp = &maxv;
        minMaxIdx(grad,minp,maxp);
        grad = (grad-minv)*255/(maxv-minv);

        cv::Mat Mask = WaterShed_(grad,markers);
        return ColorMask(Mask,basins);
    }

    ////////////without user specified markers/////////////////
    int findLocalMinimum(const cv::Mat& gray_img, cv::Mat &markers)
    /*
     * gray_img: a 1 channel img.
     * markers: minimum position marked as different positive integers,
     *          4 image edges marked as SHED,
     *          other position is 0.(output)
     *
     * return: number of minimum areas.
     */
    {
        assert(gray_img.channels() == 1);
        cv::Mat gray;
        gray_img.convertTo(gray,CV_8U);
        markers = cv::Mat::zeros(gray_img.rows,gray_img.cols,CV_32S);

        int Rows = gray.rows - 1, Cols = gray.cols - 1;

        int *p = markers.ptr<int >(0);
        for(int i = 0;i < markers.cols;i++){p[i] = SHED;}
        p = markers.ptr<int >(Rows);
        for(int i = 0;i < markers.cols;i++){p[i] = SHED;}
        for(int i = 1;i < Rows;i++)
        {
            p = markers.ptr<int >(i);
            p[0] = SHED;
            p[Cols] = SHED;
        }

        int mark = 0;
        std::queue<point2D> queue;
        for(int i = 1;i < Rows;i++)
        {
            for(int j = 1;j < Cols;j++)
            {
                //check neighbors.
                uchar val = gray.at<uchar >(i,j);
                if(markers.at<int >(i,j) == 0 && val <= gray.at<uchar >(i-1,j) && val <= gray.at<uchar >(i+1,j) && val <= gray.at<uchar >(i,j-1) && val <= gray.at<uchar >(i,j+1))
                {
                    mark++;
                    markers.at<int >(i,j) = mark;
                    queue.push({i,j});

                    while(!queue.empty())
                    {
                        point2D cur = queue.front();
                        queue.pop();

                        int R = cur.r-1, C = cur.c;
                        int neighbor_mark = markers.at<int >(R,C);
                        if(neighbor_mark == 0)
                        {
                            if(val == gray.at<uchar >(R,C))
                            {
                                markers.at<int >(R,C)=mark;
                                queue.push({R,C});
                            }
                        }

                        R = cur.r+1, C = cur.c;
                        neighbor_mark = markers.at<int >(R,C);
                        if(neighbor_mark == 0)
                        {
                            if(val == gray.at<uchar >(R,C))
                            {
                                markers.at<int >(R,C)=mark;
                                queue.push({R,C});
                            }
                        }

                        R = cur.r, C = cur.c-1;
                        neighbor_mark = markers.at<int >(R,C);
                        if(neighbor_mark == 0)
                        {
                            if(val == gray.at<uchar >(R,C))
                            {
                                markers.at<int >(R,C)=mark;
                                queue.push({R,C});
                            }
                        }

                        R = cur.r, C = cur.c+1;
                        neighbor_mark = markers.at<int >(R,C);
                        if(neighbor_mark == 0)
                        {
                            if(val == gray.at<uchar >(R,C))
                            {
                                markers.at<int >(R,C)=mark;
                                queue.push({R,C});
                            }
                        }
                    }
                }
            }
        }

        return mark;
    }

    cv::Mat WaterShed(const cv::Mat & img)
    /*
     * img: source image.
     *
     * return: watershed result.
     */
    {
        cv::Mat gray;
        cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);

        cv::Mat markers;
        int num = findLocalMinimum(gray,markers);
        std::cout << num << std::endl;

        cv::Mat gray_img;
        //cv::bilateralFilter(gray,gray_img,0,10,5); //bilateral filter
        cv::GaussianBlur(gray,gray_img,{3,3},1);
        cv::Mat g_x,g_y;
        cv::Sobel(gray_img,g_x,CV_64F,1,0);
        cv::Sobel(gray_img,g_y,CV_64F,0,1);
        cv::Mat grad = g_x.mul(g_x) + g_y.mul(g_y);
        cv::sqrt(grad,grad);
        double minv = 0.0, maxv = 0.0;
        double* minp = &minv;
        double* maxp = &maxv;
        minMaxIdx(grad,minp,maxp);
        grad = (grad-minv)*255/(maxv-minv);

        auto Mask = WaterShed_(grad,markers);
        return ColorMask(Mask,num);
    }

    void WaterShedVideo(const std::string &video_name) // read video frame, watershed it and save new video on disk.
    {
        cv::VideoCapture cap;
        cap.open(video_name);
        cv::Size size=cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter writer("./WaterShed/out.mp4",cv::VideoWriter::fourcc('X','2','6','4'),cap.get(cv::CAP_PROP_FPS),size, true);

        if(writer.isOpened())
        {
            if(cap.isOpened())
            {
                cv::Mat frame;
                while(true)
                {
                    cap>>frame;
                    if(frame.empty())
                    {
                        break;
                    }
                    auto F = WaterShed(frame);
                    cv::imshow("res",F);
                    cv::waitKey(1);
                    writer<<F;
                }
                cap.release();
                writer.release();
            }
        }
    }

}

#endif //IMAGEPROCESSINGFROMSCRATCH_WATERSHED_H
