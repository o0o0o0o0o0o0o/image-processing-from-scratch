//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_IMAGE_MATTING_H
#define IMAGEPROCESSINGFROMSCRATCH_IMAGE_MATTING_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <string>

#include "Eigen_Solver.h"

namespace ImageMatting {

    cv::SparseMat getLaplacian(const cv::Mat &I, const cv::Mat &mask, int r, double err)
/*
 * I: 3 channel image
 * mask: unknown alpha is marked as 0
 * r: window radius
 * err: absolute value less than err is considered 0
 *
 * return: L(matting Laplacian)
 */
    {
        double eps = 0.0000001f;
        int h = I.rows;
        int w = I.cols;
        int c = I.channels();
        assert(c == 3);

        int win_diam = 2 * r + 1;
        int win_size = win_diam * win_diam;

        int Lsize[2] = {h * w, h * w};
        cv::SparseMat L(2, Lsize, CV_64F);
        cv::Mat erode_mask;
        cv::erode(mask, erode_mask, cv::Mat::ones(win_diam, win_diam, CV_64F));

        for (int i = r; i < h - r; i++) {
            for (int j = r; j < w - r; j++) {
                int start_r = i - r, end_r = i + r + 1, start_c = j - r, end_c = j + r + 1;
                bool pass_win = true;
                for (int k = start_r; k < end_r; k++) {
                    for (int l = start_c; l < end_c; l++) {
                        pass_win = pass_win && (abs(erode_mask.at<double>(k, l)) > err);
                        if (!pass_win) {
                            break;
                        }
                    }
                    if (!pass_win) {
                        break;
                    }
                }
                if (!pass_win)// pass window if no unknown alpha
                {
                    cv::Vec3d mu_ij(0, 0, 0);
                    for (int k = start_r; k < end_r; k++) {
                        for (int l = start_c; l < end_c; l++) {
                            mu_ij += I.at<cv::Vec3d>(k, l);
                        }
                    }
                    mu_ij /= win_size;

                    cv::Mat sigma_ij = cv::Mat::zeros(c, c, CV_64FC1);
                    double coeff = eps / win_size;
                    for (int c1 = 0; c1 < c; c1++) {
                        for (int c2 = c1; c2 < c; c2++) {
                            double val = 0;
                            for (int k = start_r; k < end_r; k++) {
                                for (int l = start_c; l < end_c; l++) {
                                    const auto &temp_color = I.at<cv::Vec3d>(k, l);
                                    val += temp_color[c1] * temp_color[c2];
                                }
                            }
                            val /= win_size;
                            val -= mu_ij[c1] * mu_ij[c2];
                            if (c1 == c2) {
                                val += coeff;
                                sigma_ij.at<double>(c1, c2) = val;
                            } else {
                                sigma_ij.at<double>(c1, c2) = val;
                                sigma_ij.at<double>(c2, c1) = val;
                            }
                        }
                    } // covar + eps/win_size * I

                    cv::Mat sigma_ij_inv;
                    cv::invert(sigma_ij, sigma_ij_inv);

                    for (int k1 = start_r; k1 < end_r; k1++) {
                        for (int l1 = start_c; l1 < end_c; l1++) {
                            for (int k2 = k1; k2 < end_r; k2++) {
                                bool same_row = (k1 == k2);
                                for (int l2 = same_row ? l1 : start_c; l2 < end_c; l2++) {
                                    auto V1 = cv::Mat(I.at<cv::Vec3d>(k1, l1) - mu_ij).t();
                                    auto V2 = cv::Mat(I.at<cv::Vec3d>(k2, l2) - mu_ij);
                                    double val = -((1 + cv::Mat((V1 * sigma_ij_inv * V2)).at<double>(0)) / win_size);

                                    int ind1 = k1 * w + l1, ind2 = k2 * w + l2;
                                    if (same_row && l1 == l2) {
                                        val = 1 + val;
                                        *((double *) L.ptr(ind1, ind2, true)) += val;
                                    } else {
                                        *((double *) L.ptr(ind1, ind2, true)) += val;
                                        *((double *) L.ptr(ind2, ind1, true)) += val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return L;
    }

    cv::Mat Matting(const cv::Mat &Img, const cv::Mat &Img_scribble,int maxIters = INT32_MAX,double toler = TOL, int r = 1)
/*
 * Img: 3 channel original image
 * Img_scribble: 3 channel scribble image with the same size as original image
 * maxIters: max iterations when solving eqs.
 * toler: tolerance when solving eqs.
 * r: window radius
 *
 * return: alpha matte
 */
    {
        int h = Img.rows;
        int w = Img.cols;
        int c = Img.channels();
        assert(Img_scribble.rows == h && Img_scribble.cols == w && Img_scribble.channels() == c && c == 3);

        cv::Mat image, image_gray, scribble_gray;
        Img.convertTo(image, CV_64FC3, 1.0 / 255.0);   // original image
        cv::cvtColor(Img, image_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(Img_scribble, scribble_gray, cv::COLOR_BGR2GRAY);
        image_gray.convertTo(image_gray, CV_64FC1, 1.0 / 255.0);
        scribble_gray.convertTo(scribble_gray, CV_64FC1, 1.0 / 255.0);

        double err = 1e-12f;
        double lambda = 100.f;

        cv::Mat lambda_b = scribble_gray;
        cv::Mat lambda_D = cv::Mat::zeros(h, w, CV_64FC1);
        cv::Mat temp = scribble_gray - image_gray;
        for (int i = 0; i < temp.rows; i++) {
            double *temp_p = temp.ptr<double>(i);
            double *lambda_b_p = lambda_b.ptr<double>(i);
            double *lambda_D_p = lambda_D.ptr<double>(i);
            for (int j = 0; j < temp.cols; j++) {
                if (std::abs(temp_p[j]) < err) {
                    lambda_b_p[j] = 0;
                } else {
                    lambda_b_p[j] *= lambda;
                    lambda_D_p[j] = lambda;
                }
            }
        }
        temp.release();

        std::cout << "Computing Matting Laplacian." << std::endl;
        cv::SparseMat A = getLaplacian(image, lambda_D, r, err);
        std::cout << "done." << std::endl;
        cv::Mat b = lambda_b.reshape(0, h * w);
        for (int i = 0; i < h; i++) {
            double *lambda_D_p = lambda_D.ptr<double>(i);
            for (int j = 0; j < w; j++) {
                int ind = i * w + j;
                *((double *) A.ptr(ind, ind, true)) += lambda_D_p[j];
            }
        }
        lambda_b.release();
        lambda_D.release();

        // solve sparse eqs.
        cv::Mat alpha = SolveEquations(A, b,maxIters,toler);

        alpha = alpha.reshape(0, {h, w});
        for (int i = 0; i < h; i++) {
            double *alpha_p = alpha.ptr<double>(i);
            for (int j = 0; j < w; j++) {
                if (alpha_p[j] < 0) {
                    alpha_p[j] = 0;
                }
                if (alpha_p[j] > 1) {
                    alpha_p[j] = 1;
                }
            }
        }

        return alpha;
    }

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

    cv::Mat drawScribble(const cv::Mat & img, const std::string& imgname)
    /*
     * img: source image.
     *
     * return: scribble image.
     */
    {
        cv::Mat Img = img.clone();

        cv::Point CurPoint;
        bool Drawing = false;
        auto param = std::make_pair(CurPoint,Drawing);

        cv::namedWindow("Mouse Event", 0);
        cv::setMouseCallback("Mouse Event", onMouse, (void *)&param);
        int Thickness = 0;
        uchar B = 255, G = 255, R = 255;

        cv::namedWindow("slide bar", 0);
        cv::createTrackbar("thickness", "slide bar", &Thickness, 100, nullptr);

        int key;
        bool finish = false;
        while (true)
        {
            CurPoint = param.first;
            Drawing = param.second;
            if (Drawing)
            {
                circle(Img, CurPoint, 0, cv::Scalar(B, G, R), Thickness + 1);
            }
            imshow("Mouse Event", Img);
            key = cv::waitKey(1);
            switch (key)
            {
                case 27:
                    finish = true;
                    break;
                case 'c':
                case 'C':
                    Img = img.clone();
                    break;
                case 'w':
                case 'W':
                    B = 255, G = 255, R = 255;
                    break;
                case 'b':
                case 'B':
                    B = 0, G = 0, R = 0;
                    break;
                default:
                    break;
            }
            if (finish){break;}
        }
        cv::imwrite(imgname,Img);
        cv::destroyAllWindows();
        return Img;
    }

}
#endif //IMAGEPROCESSINGFROMSCRATCH_IMAGE_MATTING_H
