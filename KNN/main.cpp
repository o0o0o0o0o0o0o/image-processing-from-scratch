//
// author: 会飞的吴克
//

#include "knn.h"
#include <iostream>


void test_knn()
{
    std::vector<std::vector<double> > Data;
    double points[6][2] = {
            {2,3},{5,4},{9,6},{4,7},{8,1},{7,2}
    };
    for(unsigned long i = 0;i < 6;i++)
    {
        std::vector<double> t(points[i],points[i]+2);
        Data.push_back(t);
    }
    KNN::KDTree T(Data);
    auto neighbors = T.findKneighbor(std::vector<double>({7,4}),4);
    for(auto it = neighbors.begin();it < neighbors.end();++it)
    {
        std::cout<<"(";
        for(auto it2 = Data[(*it).first].begin();it2 < Data[(*it).first].end();++it2)
        {
            if(Data[(*it).first].end()-it2 == 1)
            {
                std::cout<<(*it2)<<") ";
            }
            else
            {
                std::cout<<(*it2)<<",";
            }
        }
        std::cout<<(*it).second;
        std::cout<<std::endl;
    }
}

int main() {
    test_knn();
    return 0;
}