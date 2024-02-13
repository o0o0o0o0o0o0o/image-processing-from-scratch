//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_PCA_H
#define IMAGEPROCESSINGFROMSCRATCH_PCA_H
namespace EigenValue{
    const double EQUAL_ERR = 1e-10;
    const double JACOBI_TOLERANCE = EQUAL_ERR;

    std::pair<std::vector<double >, std::vector<std::vector<double> > > jacobiRotation(const std::vector<std::vector<double> > & A)
    /*
     * A: a symmetric matrix.
     * output: eigen value and corresponding eigen vector(row vector)
     */
    {
        assert(!A.empty());
        for(auto it = A.begin();it < A.end();++it)
        {
            assert(it->size() == A.size());
        }             // check square matriax.
        for(unsigned long i = 0;i < A.size();i++)
        {
            for(unsigned long j = 0;j < i;j++)
            {
                assert(abs(A[i][j] - A[j][i]) <= EQUAL_ERR);
            }
        }            // check symmetry

        std::vector<std::vector<double> > A_(A);
        std::vector<std::vector<double> > P;
        for(unsigned long i = 0;i < A_.size();i++)
        {
            std::vector<double> temp;
            for(unsigned long j = 0;j < A_.size();j++)
            {
                i==j?temp.push_back(1):temp.push_back(0);
            }
            P.emplace_back(temp);
        }          // identity matrix

        while(true)
        {
            double Max_abs = 0;
            unsigned long Mi = 0,Mj = 0;
            for(unsigned long i = 0;i < A_.size();i++)
            {
                for(unsigned long j = i+1;j < A_.size();j++)
                {
                    if(abs(A_[i][j]) > Max_abs)
                    {
                        Max_abs = abs(A_[i][j]);
                        Mi = i;
                        Mj = j;
                    }
                }
            }
            if(Max_abs <= JACOBI_TOLERANCE)
            {
                break;
            } // find max absolute value each rotation, good but time consuming

            double cot2theta = (A_[Mi][Mi] - A_[Mj][Mj])/(2*A_[Mi][Mj]);
            double delta = sqrt(cot2theta*cot2theta+1);
            double t = cot2theta > 0?-cot2theta+delta:-cot2theta-delta;
            double c = 1/sqrt(1+t*t), s = c*t;

            double A_MiMi = A_[Mi][Mi]; //backup
            for(unsigned long i = 0;i < A_.size();i++)
            {
                double temp = P[Mi][i]; //backup
                P[Mi][i] = P[Mi][i]*c + P[Mj][i]*s;
                P[Mj][i] = P[Mj][i]*c - temp*s;
                for(unsigned long j = i;j < A_.size();j++)
                {
                    if(i == Mi && j == Mi)
                    {
                        A_[i][j] = A_MiMi*c*c + A_[Mj][Mj]*s*s + 2*A_[Mj][Mi]*s*c;
                    }
                    else if(i == Mi && j == Mj)
                    {
                        A_[i][j] = 0; //(A_[Mj][Mj] - A_MiMi)*s*c + A_[Mj][Mi]*(c*c - s*s);
                    }
                    else if(i == Mj && j == Mj)
                    {
                        A_[i][j] = A_MiMi*s*s + A_[Mj][Mj]*c*c - 2*A_[Mj][Mi]*s*c;
                    }
                    else if(i == Mi)
                    {
                        A_[i][j] = j < Mj?A_[j][Mi]*c + A_[Mj][j]*s:A_[i][j] = A_[j][Mi]*c + A_[j][Mj]*s;
                    }
                    else if(i == Mj)
                    {
                        A_[i][j] = A_[j][Mj]*c - A_[j][Mi]*s;
                    }
                    else if(j == Mi)
                    {
                        A_[i][j] = A_[Mi][i]*c + A_[Mj][i]*s;
                    }
                    else if(j == Mj)
                    {
                        A_[i][j] = i < Mi?A_[Mj][i]*c - A_[Mi][i]*s:A_[Mj][i]*c - A_[i][Mi]*s;
                    }
                }
            } //upper triangle (updating values of upper triangle using the "old data" in the lower triangle (since its symmetric))

//            double norm_Mi = 0, norm_Mj = 0;
//            for(unsigned long i = 0;i < P.size();i++)
//            {
//                norm_Mi += P[Mi][i] * P[Mi][i];
//                norm_Mj += P[Mj][i] * P[Mj][i];
//            }
//            norm_Mi = sqrt(norm_Mi);
//            norm_Mj = sqrt(norm_Mj);
//            for(unsigned long i = 0;i < P.size();i++)
//            {
//                P[Mi][i] /= norm_Mi;
//                P[Mj][i] /= norm_Mj;
//            }
            //normalization(for stability consideration.)

            for(unsigned long j = 0;j < Mi;j++) A_[Mi][j] = A_[j][Mi];
            for(unsigned long j = 0;j < Mj;j++) A_[Mj][j] = A_[j][Mj];
            for(unsigned long i = Mi+1;i < A_.size();i++) A_[i][Mi] = A_[Mi][i];
            for(unsigned long i = Mj+1;i < A_.size();i++) A_[i][Mj] = A_[Mj][i]; //lower triangle
        }

        std::vector<double > eignValue;
        for(unsigned long i = 0;i < A_.size();i++) eignValue.push_back(A_[i][i]);
        return std::make_pair(eignValue,P);
    }

    std::vector<std::vector<double > > multiply(const std::vector<std::vector<double > > & A, const std::vector<std::vector<double > > & B)
    /*
     * A: a nxk matrix
     * B: a kxm matrix
     * output: AxB
     */
    {
        assert(!A.empty());
        assert(!B.empty());
        for(unsigned long i = 1;i < A.size();i++)
        {
            assert(A[i].size() == A[0].size());
        }
        assert(A[0].size() == B.size());
        for(unsigned long i = 1;i < B.size();i++)
        {
            assert(B[i].size() == B[0].size());
        }

        std::vector<std::vector<double > > result;
        for(unsigned long i = 0;i < A.size();i++)
        {
            std::vector<double > temp;
            for(unsigned long j = 0;j < B[0].size();j++)
            {
                double dotproduct = 0;
                for(unsigned long k = 0;k < B.size();k++)
                {
                    dotproduct += A[i][k] * B[k][j];
                }
                temp.push_back(dotproduct);
            }
            result.emplace_back(temp);
        }

        return result;
    }

    std::vector<std::vector<double > > transpose(const std::vector<std::vector<double > > & A)
    /*
     * A: a matrix
     * output: A's transpose
     */
    {
        assert(!A.empty());
        for(unsigned long i = 1;i < A.size();i++)
        {
            assert(A[i].size() == A[0].size());
        }

        std::vector<std::vector<double > > result;
        for(unsigned long i = 0;i < A[0].size();i++)
        {
            std::vector<double > temp;
            for(unsigned long j = 0;j < A.size();j++)
            {
                temp.push_back(A[j][i]);
            }
            result.emplace_back(temp);
        }

        return result;
    }

    std::pair<std::pair<std::vector<double >, std::vector<std::vector<double > > >,std::vector<std::vector<double > > >
            PCA(const std::vector<std::vector<double> > & A, unsigned long k, double lambda_proportion = 0.95)
            /*
             * A: orignal data(each row is considered a sample, each column is considered a feature).
             * k: the dimension of the compressed data(K <= A.columns) if k <= 0, use lambda proportion to judge( 1 >= lambda proportion > 0)
             * output: < <mean of samples, new axis vectors>, compressed data>
             */
    {
        assert(!A.empty());
        assert(k <= A[0].size());
        if(k <= 0)
            assert(lambda_proportion>0&&lambda_proportion<=1);

        for(unsigned long i = 1;i < A.size();i++)
        {
            assert(A[i].size() == A[0].size());
        }

        std::vector<double > mean = A[0];
        for(unsigned long i = 1;i < A.size();i++)
        {
            for(unsigned long j = 0;j < A[i].size();j++)
            {
                mean[j] += A[i][j];
            }
        }
        for(unsigned long i = 0;i < mean.size();i++)
        {
            mean[i] /= A.size();
        }
        // calculate mean.

        std::vector<std::vector<double > > A_(A);
        for(unsigned long i = 0;i < A_.size();i++)
        {
            for(unsigned long j = 0;j < A_[i].size();j++)
            {
                A_[i][j] -= mean[j];
            }
        }
        // move data center to the origin.

        auto conv = multiply(transpose(A_),A_);
//        std::cout<<"jacobi"<<std::endl;
        auto eigen = jacobiRotation(conv);
//        std::cout<<"jacobi done"<<std::endl;
        std::vector<std::pair<double , std::vector<double > > > eigen_;
        for(unsigned long i = 0;i < eigen.first.size();i++)
        {
            eigen_.emplace_back(eigen.first[i],eigen.second[i]);
        }

        sort(eigen_.begin(),eigen_.end(),
                [](const std::pair<double , std::vector<double > > & e1,
                                            const std::pair<double , std::vector<double > > & e2)
                                            {
                                                return e1.first > e2.first;
                                            }
        );


        std::vector<std::vector<double > > eigen_vec;
        if(k > 0)
        {
            for(unsigned long i = 0;i < k;i++)
            {
                eigen_vec.push_back(eigen_[i].second);
            }
        }
        else
        {
            double lambda_sum = 0;
            for(unsigned long i = 0;i < eigen_.size();i++)
            {
                lambda_sum += eigen_[i].first;
            }
            double cur_sum = 0;
            for(unsigned long i = 0;cur_sum/lambda_sum < lambda_proportion&&i < eigen_.size();i++)
            {
                cur_sum += eigen_[i].first;
                eigen_vec.push_back(eigen_[i].second);
            }
        }

        std::vector<std::vector<double > > new_data = multiply(A_,transpose(eigen_vec));

        return std::make_pair(std::make_pair(mean,eigen_vec),new_data);
    }

}
#endif //IMAGEPROCESSINGFROMSCRATCH_PCA_H
