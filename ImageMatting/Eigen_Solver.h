//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_EIGEN_SOLVER_H
#define IMAGEPROCESSINGFROMSCRATCH_EIGEN_SOLVER_H
#define TOL 2.5e-10

#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>

namespace ImageMatting {

    cv::Mat SolveEquations(const cv::SparseMat &A, const cv::Mat &b,int maxIters = INT32_MAX,double toler = TOL)
/*
 * A: coefficient matrix
 * b: result vector
 *
 * return: solution vector
 */
    {
        // convert to eigen matrix
        typedef Eigen::Triplet<double> Trip;
        std::vector<Trip> trp;
        for (auto it = A.begin(); it != A.end(); it++) {
            auto node = it.node();
            trp.emplace_back(node->idx[0], node->idx[1], it.value<double>());
        }
        int dim = A.size()[0];
        Eigen::SparseMatrix<double> AES(dim, dim);
        AES.setFromTriplets(trp.begin(), trp.end());

        Eigen::VectorXd bE;
        cv::cv2eigen(b, bE);


        //solve (L+lambda*D)alpha = lambda*b

        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper|Eigen::Lower> solver;
        solver.setMaxIterations(maxIters);
        solver.setTolerance(toler);
        std::cout << "Preconditioning." << std::endl;
        solver.compute(AES);
        std::cout << "done." << std::endl;
        std::cout << "Solving equations." << std::endl;
        Eigen::VectorXd x = solver.solve(bE);
        std::cout << "done." << std::endl;

        cv::Mat alpha;
        cv::eigen2cv(x, alpha);

        return alpha;
    }
}

#endif //IMAGEPROCESSINGFROMSCRATCH_EIGEN_SOLVER_H
