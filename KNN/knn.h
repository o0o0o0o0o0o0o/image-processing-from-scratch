//
// author: 会飞的吴克
//


#ifndef IMAGEPROCESSINGFROMSCRATCH_KNN_H
#define IMAGEPROCESSINGFROMSCRATCH_KNN_H
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

namespace KNN
{
    class KDTree
    {
        private:
            class treenode{
                public:
                    treenode(const std::vector<double > & coord, const unsigned long & axis, const unsigned long & ind):
                            axis(axis),index(ind),lchild(nullptr),rchild(nullptr)
                    {
                        assert(!coord.empty());
                        coordinate = new double[coord.size()];
                        for (unsigned long i= 0; i < coord.size(); ++i) {
                            coordinate[i] = coord[i];
                        }
                    };
                    ~treenode(){
                        delete [] coordinate;
                    }

                    double* getCoordinate() const { return coordinate;}
                    unsigned long getAxis() const { return axis;}
                    unsigned long getIndex() const { return index;}
                    treenode* getLchild() const { return lchild;}
                    treenode* getRchild() const { return rchild;}
                    void setLchild(treenode* t){lchild = t;}
                    void setRchild(treenode* t){rchild = t;}
                private:
                    double *coordinate;
                    unsigned long axis;
                    unsigned long index;
                    treenode *lchild, *rchild;
            }; // tree node.

            treenode * buildKDTree(const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator &,
                    const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator &);

            static void deleteKDTree(treenode *);

            static double Variance(const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator &,
                            const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator &,
                            const unsigned long &);
            void findPath(std::vector<treenode* > &, treenode*,const std::vector<double> &) const;

            static void orderInsert(std::vector<std::pair<unsigned long,double> > &,const std::pair<unsigned long,double> &);

            treenode * tree;
            unsigned long dim;
        public:
            explicit KDTree(const std::vector<std::vector<double > > &);
            ~KDTree();
            std::vector<std::pair<unsigned long,double> > findKneighbor(const std::vector<double > &,const unsigned long &) const;
    };

    KDTree::KDTree(const std::vector<std::vector<double> > & Data)
    {
        assert(!Data.empty());
        assert(!Data[0].empty());
        std::vector< std::pair<std::vector<double >, unsigned long > > DataWithInd;
        DataWithInd.push_back(std::make_pair(Data[0],0));
        for(unsigned long i = 1;i < Data.size();i++)
        {
            assert(Data[i].size() == Data[0].size());
            DataWithInd.push_back(std::make_pair(Data[i],i));
        }
        dim = Data[0].size();
        tree = buildKDTree(DataWithInd.begin(),DataWithInd.end());
    } // build the tree.

    KDTree::treenode* KDTree::buildKDTree(const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator & l,
                                          const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator & r)
    {
        unsigned long Size = r - l;
        if(Size > 1)
        {
            unsigned long axis = 0;
            double max_var = Variance(l,r,axis);
            for(unsigned long i = 1;i < dim;i++)
            {
                double temp = Variance(l,r,i);
                if(temp > max_var)
                {
                    max_var = temp;
                    axis = i;
                }
            }

            std::sort(l,r,
                    [&axis]
                    (const std::pair<std::vector<double >, unsigned long > & x,
                            const std::pair<std::vector<double >, unsigned long > & y)
                    {
                        return (x.first)[axis] < (y.first)[axis];
                    }
                            );

            auto element = (l+Size/2);
            treenode* t = new treenode((*element).first,axis,(*element).second);
            t->setLchild(buildKDTree(l,element));
            t->setRchild(buildKDTree(element+1,r));
            return t;
        }
        else if(Size == 1)
        {
            treenode* t = new treenode((*l).first,((*l).first).size(),(*l).second);
            return t;
        }
        else
        {
            return nullptr;
        }
    }

    void KDTree::deleteKDTree(KNN::KDTree::treenode * t) {
        assert(t);
        auto lchild = t->getLchild();
        auto rchild = t->getRchild();
        if(lchild)
            deleteKDTree(lchild);
        //std::cout<<t->getIndex()<<" ";
        delete t;
        if(rchild)
            deleteKDTree(rchild);
    }

    double KDTree::Variance(const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator & l,
                            const std::vector< std::pair<std::vector<double >, unsigned long > >::iterator & r,
                            const unsigned long & axis)
    {
         double aver = 0;
         unsigned long N = r - l;
         for(auto i = l;i < r;i++)
         {
             aver += ((*i).first)[axis];
         }
         aver /= N;

         double sq_diff = 0;
         for(auto i = l;i < r;i++)
         {
             double temp = ((*i).first)[axis] - aver;
             sq_diff += temp*temp;
         }

        return sq_diff/(N-1);
    }

    KDTree::~KDTree() {
        deleteKDTree(this->tree);
    }

    void KDTree::findPath(std::vector<treenode* > & path,treenode* t, const std::vector<double> & P) const {
        while(t)
        {
            path.push_back(t);
            unsigned long axis = t->getAxis();
            const double * coord = t->getCoordinate();
            if(axis<dim)
                t = (P[axis] < coord[axis]?t->getLchild():t->getRchild());
            else
                t = nullptr;
        }
    }

    void KDTree::orderInsert(std::vector<std::pair<unsigned long, double> > & sorted_list,
                             const std::pair<unsigned long, double> & new_element) {
        auto l = sorted_list.begin(), r = sorted_list.end();
        unsigned long num = r-l;
        while(num)
        {
            if(new_element.second < (*(l+num/2)).second)
            {
                r = l+num/2;
            }
            else
            {
                l = l+ num/2 + 1;
            }
            num = r - l;
        }
        sorted_list.insert(r,new_element);
    }

    std::vector<std::pair<unsigned long,double> > KDTree::findKneighbor(const std::vector<double> & P,const unsigned long & k) const {
        assert(P.size() == dim);
        assert(k>0);
        std::vector<treenode* > path;
        std::vector<std::pair<unsigned long,double> > result;
        findPath(path,tree,P);
        while(!path.empty())
        {
            treenode* temp = path[path.size()-1];
            unsigned long axis = temp->getAxis();
            double* coord = temp->getCoordinate();
            if(axis >= dim) // leaf
            {
                double distance_sq = 0;
                unsigned long i = 0;
                for(auto it = P.begin();it < P.end(); ++it,++i)
                {
                    double diff = (*it) - coord[i];
                    distance_sq += diff*diff;
                }
                if(result.size() < k||distance_sq<result[result.size()-1].second)
                {
                    auto neighbor = std::make_pair(temp->getIndex(),distance_sq);
                    orderInsert(result,neighbor);
                }
                if(result.size() > k)
                {
                    result.pop_back();
                }
                path.pop_back();
            }
            else // not leaf
            {
                if(result.size() < k)
                {
                    double distance_sq = 0;
                    unsigned long i = 0;
                    for(auto it = P.begin();it < P.end(); ++it,++i)
                    {
                        double diff = (*it) - coord[i];
                        distance_sq += diff*diff;
                    }
                    orderInsert(result,std::make_pair(temp->getIndex(),distance_sq));
                    path.pop_back();
                    findPath(path,P[axis] < coord[axis]?temp->getRchild():temp->getLchild(),P);
                }
                else if(result[result.size()-1].second >= (P[axis]-coord[axis])*(P[axis]-coord[axis]))
                {
                    double distance_sq = 0;
                    unsigned long i = 0;
                    for(auto it = P.begin();it < P.end(); ++it,++i)
                    {
                        double diff = (*it) - coord[i];
                        distance_sq += diff*diff;
                    }
                    if(distance_sq < result[result.size()-1].second)
                    {
                        orderInsert(result,std::make_pair(temp->getIndex(),distance_sq));
                        result.pop_back();
                    }
                    path.pop_back();
                    findPath(path,P[axis] < coord[axis]?temp->getRchild():temp->getLchild(),P);
                }
                else
                {
                    path.pop_back();
                }
            }
        }
        return result;
    }
}

#endif //IMAGEPROCESSINGFROMSCRATCH_KNN_H