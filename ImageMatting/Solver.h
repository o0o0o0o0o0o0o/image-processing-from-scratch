//
// author: 会飞的吴克
//
#ifndef IMAGEPROCESSINGFROMSCRATCH_SOLVER_H
#define IMAGEPROCESSINGFROMSCRATCH_SOLVER_H
#include <limits>
#include <algorithm>
#define TOL 2.5e-10

namespace ImageMatting {

    class MySparseMatrix{
    public:
        explicit MySparseMatrix(const cv::SparseMat& Mat){
            assert(Mat.dims() == 2);
            auto M = Mat.size()[0];
            std::vector<std::vector<double > > values(M);
            ColIndexs.resize(M);

            int m = 0,n = 0;
            for(auto it = Mat.begin();it != Mat.end();it++)
            {
                const cv::SparseMat::Node* node = it.node();
                m = node->idx[0];
                n = node->idx[1];
                values[m].push_back(it.value<double >());
                ColIndexs[m].push_back(n);
            }
            SparseMatRows.reserve(M);
            for(int i = 0;i < M;i++)
            {
                SparseMatRows.emplace_back(cv::Mat(values[i], true));
            }
        }

        cv::Mat Multiply(const cv::Mat &b) const
        {
            assert(b.cols == 1);
            int M = SparseMatRows.size();
            int K = b.cols;
            cv::Mat res = cv::Mat::zeros(M,K,CV_64F);

            auto *b_ = new double [b.rows];
            for(int i = 0;i < b.rows;i++){b_[i] = b.at<double >(i);}
            std::vector<std::vector<double > > B(M);

            for(int i = 0;i < M;i++)
            {
                B[i].resize(ColIndexs[i].size());
                for(int j = 0;j < ColIndexs[i].size();j++)
                {
                    B[i][j] = b_[ColIndexs[i][j]];
                }
            }
            delete [] b_;

            for(int i = 0;i < M;i++)
            {
                res.at<double >(i) = SparseMatRows[i].dot(cv::Mat(B[i]));
            }
            return res;
        }

    private:
        std::vector<cv::Mat> SparseMatRows;
        std::vector< std::vector<int> > ColIndexs;
    };

    cv::Mat SolveEquations(const cv::SparseMat &A, const cv::Mat &b,int maxIters = INT32_MAX,double toler = TOL)
/*
 * note that this function only works for symmetric and positive define A.
 *
 * A: coefficient matrix
 * b: result vector
 * maxIters: max iterations.
 * toler: relative error tolerance.
 *
 * return: solution vector
 */
    {
        auto A_size = A.size();
        int A_dims = A.dims();
        assert(A_size[0] == A_size[1]);
        assert(A_dims == 2);
        assert(b.cols == 1);


        using std::abs;
        using std::sqrt;
        double zero = 2.22507e-308;     // precision.

        std::cout << "Preconditioning." << std::endl;
        // Diagonal preconditioning.
        cv::SparseMat A_;
        A.convertTo(A_,CV_64F);

        cv::Mat M_inv(A_size[0],1,CV_64F);
        double element = 0;
        for(int i = 0;i < A_size[0];i++)
        {
            element = A_.value<double >(i,i);
            M_inv.at<double >(i) = abs(element) < zero ? 1 : 1/element;
        }
        std::cout << "done." << std::endl;


        std::cout << "Solving equations." << std::endl;
        // conjugate gradient
        cv::Mat x = cv::Mat::zeros(b.rows,b.cols,CV_64F);     // initial guess.
        auto A_S = MySparseMatrix(A_);
        cv::Mat residual = b - A_S.Multiply(x); //initial residual

        int Iter_num = 0;
        double b_squaredNorm = b.dot(b);
        if(b_squaredNorm == 0)
        {
            x = cv::Mat::zeros(b.rows,b.cols,CV_64F);
        }
        else
        {
            cv::Mat d = M_inv.mul(residual);
            double deltaNew = residual.dot(d);
            double threshold = std::max(toler * toler * deltaNew, zero);
            double residual_squaredNorm = residual.dot(residual);
            if (residual_squaredNorm >= threshold)
            {
                cv::Mat s, q;
                double alpha = 0, deltaOld = 0, beta = 0;
                int i = 0;
                while(i < maxIters)
                {
                    q = A_S.Multiply(d);

                    alpha = deltaNew / d.dot(q);
                    x += alpha * d;
                    residual -= alpha * q;

                    residual_squaredNorm = residual.dot(residual);
                    if(residual_squaredNorm < threshold)
                        break;

                    s = M_inv.mul(residual);

                    deltaOld = deltaNew;
                    deltaNew = residual.dot(s);
                    beta = deltaNew / deltaOld;
                    d = s + beta * d;
                    i++;
                    std::cout << "iteration: "<< i << std::endl;
                }
                Iter_num = i;
            }
        }
        std::cout << Iter_num << " iterations totally." << std::endl;
        std::cout << "done." << std::endl;
        return x;
    }

}

#endif //IMAGEPROCESSINGFROMSCRATCH_SOLVER_H
