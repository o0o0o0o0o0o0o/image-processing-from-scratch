//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_SVD_H
#define IMAGEPROCESSINGFROMSCRATCH_SVD_H
namespace EigenValue{
    std::pair<std::vector<double >,std::pair<std::vector<std::vector<double> >, std::vector<std::vector<double> > > >
            SVD(const std::vector<std::vector<double> > & A)
    /*
     * A:the matrix to be decomposed.
     * return: <S, <U,V> >left singular vector matrix U, vector S that containing singular values, right singular vector matrix V. A = U*diag(S)*V'
     */
    {
        assert(!A.empty());
        for(unsigned long i = 1;i < A.size();i++)
        {
            assert(A[i].size() == A[0].size());
        } //check matrix

        std::vector<std::vector<double> > ATA_or_AAT;
        bool flag = A[0].size() < A.size();
        ATA_or_AAT = flag?multiply(transpose(A),A):multiply(A,transpose(A));
        auto eigen = jacobiRotation(ATA_or_AAT);
        auto S = eigen.first;
        auto V_or_U = transpose(eigen.second);
        for(auto it = S.begin();it < S.end();it++)
        {
            *it = sqrt(*it);
        }
        std::vector<std::vector<double> > U_or_V;
        for(unsigned long i = 0;i < (flag?A.size():A[0].size());i++)
        {
            std::vector<double > temp;
            for(unsigned long j = 0;j < V_or_U[0].size();j++)
            {
                double sum = 0;
                for(unsigned long k = 0;k < V_or_U.size();k++)
                {
                    sum += (flag?A[i][k]:A[k][i])*V_or_U[k][j];
                }
                temp.push_back(sum/S[j]);
            }
            U_or_V.push_back(temp);
        }

        return flag?std::make_pair(S,std::make_pair(U_or_V,V_or_U)):std::make_pair(S,std::make_pair(V_or_U,U_or_V));
    }
}
#endif //IMAGEPROCESSINGFROMSCRATCH_SVD_H
