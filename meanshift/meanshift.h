//
// author: 会飞的吴克
//

#ifndef IMAGEPROCESSINGFROMSCRATCH_MEANSHIFT_H
#define IMAGEPROCESSINGFROMSCRATCH_MEANSHIFT_H
#include <iostream>
#include <vector>
#include <algorithm>

namespace Segmentation{

    const double PI  = 3.14159265358979323846264338327950288;
    const int channels = 3;
    const double EQUAL_ERR = 1e-10;
    struct point2D_UL{
        unsigned long x,y;
    };
    class Color{
    private:
        double vals[channels]{};
    public:
        double operator[](const int i) const
        {
            return vals[i];
        }

        double& operator[](const int i)
        {
            return vals[i];
        }
        Color()= default;
        Color(double a, double b,double c){
            vals[0] = a;
            vals[1] = b;
            vals[2] = c;
        }
    };

    typedef std::vector<std::vector<Color> > Image;
    typedef std::vector<Color> Row;

    void RGBtoLUV(const Color & color, Color & color_LUV)
    {
        Color WhiteReference_XYZ = {95.047,100.000,108.883};
        Color color_XYZ;
        double r = color[0] / 255.0, g = color[1] / 255.0, b = color[2] / 255.0;
        r = (r > 0.04045 ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92) * 100.0;
        g = (g > 0.04045 ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92) * 100.0;
        b = (b > 0.04045 ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92) * 100.0;

        // Observer. = 2°, Illuminant = D65
        color_XYZ[0] = r * 0.4124 + g * 0.3576 + b * 0.1805;
        color_XYZ[1] = r * 0.2126 + g * 0.7152 + b * 0.0722;
        color_XYZ[2] = r * 0.0193 + g * 0.1192 + b * 0.9505;

        const double Epsilon = 0.008856; // Intent is 216/24389
        const double Kappa = 903.3; // Intent is 24389/27
        double y = color_XYZ[1] / WhiteReference_XYZ[1];
        color_LUV[0] = (y > Epsilon ? 116.0 * pow(y,1.0/3.0) - 16.0 : Kappa * y);

        double targetDenominator = color_XYZ[0] + color_XYZ[1]*15.0 + color_XYZ[2]*3.0;
        double referenceDenominator = WhiteReference_XYZ[0] + WhiteReference_XYZ[1]*15.0 + WhiteReference_XYZ[2]*3.0;
        // ReSharper disable CompareOfFloatsByEqualityOperator
        double xTarget = targetDenominator == 0 ? 0 : ((4.0 * color_XYZ[0] / targetDenominator) - (4.0 * WhiteReference_XYZ[0] / referenceDenominator));
        double yTarget = targetDenominator == 0 ? 0 : ((9.0 * color_XYZ[1] / targetDenominator) - (9.0 * WhiteReference_XYZ[1] / referenceDenominator));
        // ReSharper restore CompareOfFloatsByEqualityOperator

        color_LUV[1] = (13.0 * color_LUV[0] * xTarget);
        color_LUV[2] = (13.0 * color_LUV[0] * yTarget);
    }

    void RGBimage2LUVimage(const Image & RGB,Image & LUV)
    {
        assert(!RGB.empty());
        for(unsigned long i = 1;i < RGB.size();i++)
        {
            assert(RGB[i].size() == RGB[0].size());
        }

        LUV.clear();
        for(auto it = RGB.begin();it < RGB.end();it++)
        {
            Row tempRow;
            for(auto it2 = it->begin();it2 < it->end();it2++)
            {
                Color temp;
                RGBtoLUV((*it2),temp);
                tempRow.push_back(temp);
            }
            LUV.push_back(tempRow);
        }
    }

    void LUVtoRGB(const Color & luv, Color & rgb)
    {
        Color WhiteReference_XYZ = {95.047,100.000,108.883};
        const double Epsilon = 0.008856; // Intent is 216/24389
        const double Kappa = 903.3; // Intent is 24389/27

        const double c = -1.0 / 3.0;
        double Denominator = WhiteReference_XYZ[0] + 15.0 * WhiteReference_XYZ[1] + 3.0 * WhiteReference_XYZ[2];
        double uPrime = (4.0 * WhiteReference_XYZ[0]) / Denominator;
        double vPrime = (9.0 * WhiteReference_XYZ[1]) / Denominator;
        double a = (1.0 / 3.0) * ((52.0 * luv[0]) / (luv[1] + 13 * luv[0] * uPrime) - 1.0);
        double imteL_16_116 = (luv[0] + 16.0) / 116.0;
        double y = luv[0] > Kappa * Epsilon
                ? imteL_16_116 * imteL_16_116 * imteL_16_116
                : luv[0] / Kappa;
        double b = -5.0 * y;
        double d = y * ((39.0 * luv[0]) / (luv[2] + 13.0 * luv[0] * vPrime) - 5.0);
        double x = (d - b) / (a - c);
        double z = x * a + b;
        // Color color_XYZ(100 * x,100 * y,100 * z);

        double r = x * 3.2406 + y * -1.5372 + z * -0.4986;
        double g = x * -0.9689 + y * 1.8758 + z * 0.0415;
        b = x * 0.0557 + y * -0.2040 + z * 1.0570;

        r = r > 0.0031308 ? 1.055 * pow(r, 1 / 2.4) - 0.055 : 12.92 * r;
        g = g > 0.0031308 ? 1.055 * pow(g, 1 / 2.4) - 0.055 : 12.92 * g;
        b = b > 0.0031308 ? 1.055 * pow(b, 1 / 2.4) - 0.055 : 12.92 * b;

        r *= 255.0;
        g *= 255.0;
        b *= 255.0;

        rgb[0] = r < 0 ? 0 :(r > 255 ? 255 : r);
        rgb[1] = g < 0 ? 0 :(g > 255 ? 255 : g);
        rgb[2] = b < 0 ? 0 :(b > 255 ? 255 : b);
    }

    void LUVimage2RGBimage(const Image & LUV,Image & RGB)
    {
        assert(!LUV.empty());
        for(unsigned long i = 1;i < LUV.size();i++)
        {
            assert(LUV[i].size() == LUV[0].size());
        }

        RGB.clear();
        for(auto it = LUV.begin();it < LUV.end();it++)
        {
            Row tempRow;
            for(auto it2 = it->begin();it2 < it->end();it2++)
            {
                Color temp;
                LUVtoRGB((*it2),temp);
                tempRow.push_back(temp);
            }
            RGB.push_back(tempRow);
        }
    }

    double Guass(double x){
        double exponent = -x*x/2.0;
        return (1.0/(sqrt(2*PI)))*exp(exponent);
    }

    void meanshift_filter(const Image & src,Image & dst, const double sr = 6,const double tr = 8){
        const unsigned long max_itr = 10;
        const double color_threshold = 0.05;
        assert(!src.empty());
        for(unsigned long i = 0;i < src.size();i++)
        {
            assert(src[i].size() == src[0].size());
        }
        assert(sr > 0 && tr > 0);

        dst.clear();

        for(unsigned long i = 0;i < src.size();i++)
        {
            Row dst_row;
            for(unsigned long j = 0;j < src[i].size();j++)
            {
                double cur_row = i;
                double cur_col = j;
                Color cur_color = src[i][j];
                for(unsigned long k = 0;k < max_itr;k++)
                {
                    double ltr = (cur_row - sr - 1);// left top row.
                    double ltc = (cur_col - sr - 1);// left top col.
                    double rbr = (cur_row + sr + 1);// right bottom row.
                    double rbc = (cur_col + sr + 1);// right bottom col.

                    ltr = ltr < 0 ? 0 : ltr;
                    ltc = ltc < 0 ? 0 : ltc;
                    rbr = rbr > src.size() ? src.size() : rbr;
                    rbc = rbc > src[i].size() ? src[i].size() : rbc;

                    double displacement_r = 0;
                    double displacement_c = 0;
                    double denominator = 0;
                    for(unsigned long l = ltr;l < rbr;l++)
                    {
                        for(unsigned long m = ltc;m < rbc;m++)
                        {
                            double temp = cur_row - l;
                            double distant_sq = 0;
                            distant_sq += (temp*temp);
                            temp = cur_col - m;
                            distant_sq += (temp*temp);

                            if(distant_sq <= sr*sr)
                            {
                                double color_distance = 0;
                                for(int c = 0;c < channels;c++)
                                {
                                    temp = src[l][m][c] - cur_color[c];
                                    color_distance += (temp*temp);
                                }
                                color_distance = sqrt(color_distance);

                                double weight = Guass(color_distance/tr);
                                displacement_r += (l - cur_row) * weight;
                                displacement_c += (m - cur_col) * weight;
                                denominator += weight;
                            }
                        }
                    }

                    double dr = displacement_r / denominator;
                    double dc = displacement_c / denominator;
                    cur_row += dr;
                    cur_col += dc;
                    Color pre_color = cur_color;

                    if(cur_row < 0 || cur_row >= src.size() - 1 || cur_col < 0 || cur_col >= src[i].size() - 1)
                    {
                        int cur_r = cur_row;
                        int cur_c = cur_col;
                        cur_r = cur_r < 0 ? 0 : cur_r;
                        cur_c = cur_c < 0 ? 0 : cur_c;
                        cur_r = cur_r >= src.size() ? src.size() - 1 : cur_r;
                        cur_c = cur_c >= src[i].size() ? src[i].size() - 1 : cur_c;
                        cur_color[0] = src[cur_r][cur_c][0];
                        cur_color[1] = src[cur_r][cur_c][1];
                        cur_color[2] = src[cur_r][cur_c][2];
                    }
                    else
                    {
                        for(int c = 0;c < channels;c++)
                        {
                            int lt_r = cur_row,lt_c = cur_col;
                            double temp1 = src[lt_r][lt_c][c] + (cur_col - lt_c)*(src[lt_r][lt_c+1][c] - src[lt_r][lt_c][c]);
                            double temp2 = src[lt_r+1][lt_c][c] + (cur_col - lt_c)*(src[lt_r+1][lt_c+1][c] - src[lt_r+1][lt_c][c]);
                            cur_color[c] = temp1 + (cur_row - lt_r)*(temp2-temp1);
                        }
                    }
                    // bilinear interpolation

                    double color_distance = 0;
                    for(int c = 0;c < channels;c++)
                    {
                        double temp = pre_color[c] - cur_color[c];
                        color_distance += (temp*temp);
                    }

                    if((abs(dr) < EQUAL_ERR && abs(dc) < EQUAL_ERR) || color_distance < color_threshold*color_threshold ||k == max_itr - 1)
                    {
                        dst_row.push_back(cur_color);
                        // you can record the converge point here (cur_row,cur_col)
                        break;
                    }
                }
            }
            dst.push_back(dst_row);
        }
    }

    void meanshift_segmentation(const Image & src,Image & dst,const int minarea = 20,const double sr = 6,const double tr = 8){
        assert(sr > 0 && tr > 0 && minarea >= 0);
        assert(!src.empty());
        for(unsigned long i = 0;i < src.size();i++)
        {
            assert(src[i].size() == src[0].size());
        }
        dst.clear();

        Image filt;
        meanshift_filter(src,filt,sr,tr);

        Row Colors;
        std::vector<int> pointnum;

        std::vector<std::vector<int> > Label;
        Label.reserve(filt.size());
        std::vector<int> temprow;
        temprow.reserve(filt[0].size());
        for(unsigned long i=0;i < filt[0].size();i++){temprow.push_back(0);}
        for(unsigned long i=0;i < filt.size();i++){Label.push_back(temprow);}

        std::stack<point2D_UL> S;
        Color curcolor, average;
        int region = 0;
        int num = 0;

        int displacement[8][2] = {
                {-1,-1},{-1,0},{-1,1},
                {0,-1},        {0,1},
                {1,-1}, {1,0}, {1,1}
        };
        // 8-neighbors
        double tr_sq = tr * tr;

        ///////////// build Label ///////////////////
        for(unsigned long i = 0;i < filt.size();i++)
        {
            for(unsigned long j = 0;j < filt[i].size();j++)
            {
                if(Label[i][j] <= 0)
                {
                    region++;
                    Label[i][j] = region;
                    S.push({i,j});
                    curcolor = filt[i][j];
                    num = 1;
                    average = curcolor;

                    while(true)
                    {
                        if(S.empty()){break;}
                        point2D_UL curpos = S.top();
                        S.pop();

                        for(int k = 0;k < 8;k++)
                        {
                            unsigned long r = curpos.x + displacement[k][0];
                            unsigned long c = curpos.y + displacement[k][1];
                            if(r >= 0 && r < filt.size() && c >= 0 && c < filt[0].size() && Label[r][c] <= 0)
                            {
                                Color cl = filt[r][c];
                                double color_distance_sq = 0;
                                for(int channel = 0;channel < channels;channel++)
                                {
                                    double temp = curcolor[channel] - cl[channel];
                                    color_distance_sq += (temp*temp);
                                }
                                if(color_distance_sq <= tr_sq)
                                {
                                    Label[r][c] = region;
                                    S.push({r,c});
                                    num++;

                                    for(int channel = 0;channel < channels;channel++)
                                    {
                                        average[channel] += cl[channel];
                                    }
                                }
                            }
                        }
                    }

                    Colors.emplace_back(average[0]/num,average[1]/num,average[2]/num);
                    pointnum.push_back(num);
                }
            }
        }

        /////////////////////// TransitiveClosure ////////////////////////////
        int max_iter = 5;

        // build adjoint table
        std::vector<std::vector<int> > adjoint_table;
        adjoint_table.reserve(region);
        for(int i = 0;i < region;i++){adjoint_table.emplace_back();}
        std::vector<int> adjoint_table_head;
        adjoint_table_head.reserve(region);
        for(int i = 0;i < region;i++){adjoint_table_head.push_back(i+1);}

        int L1 = 0,L2 = 0;
        for(unsigned long i = 0;i < Label.size();i++)
        {
            for(unsigned long j = 0;j < Label[i].size();j++)
            {
                if(i > 0 && Label[i][j] != Label[i-1][j])
                {
                    L1 = Label[i][j],L2 = Label[i-1][j];
                    std::vector<int>::iterator it = std::find(adjoint_table[L1-1].begin(),adjoint_table[L1-1].end(),L2);
                    if(adjoint_table[L1-1].end()==it)
                    {
                        adjoint_table[L1-1].push_back(L2);
                        adjoint_table[L2-1].push_back(L1);
                    }
                }
                if(j > 0 && Label[i][j] != Label[i][j-1])
                {
                    L1 = Label[i][j],L2 = Label[i][j-1];
                    std::vector<int>::iterator it = std::find(adjoint_table[L1-1].begin(),adjoint_table[L1-1].end(),L2);
                    if(adjoint_table[L1-1].end()==it)
                    {
                        adjoint_table[L1-1].push_back(L2);
                        adjoint_table[L2-1].push_back(L1);
                    }
                }
            }
        }


        // merge similar areas.
        for(int i = 0,deltaRegion = 1;i < max_iter && deltaRegion > 0;i++)
        {
            for(unsigned long r = 0;r < adjoint_table.size();r++)
            {
                for(unsigned long c = 0;c < adjoint_table[r].size();c++)
                {
                    double color_distance_sq = 0;
                    Color color1 = Colors[r], color2 = Colors[adjoint_table[r][c]-1];
                    for(int channel = 0;channel < channels;channel++)
                    {
                        double temp = color1[channel] - color2[channel];
                        color_distance_sq += (temp*temp);
                    }

                    if(color_distance_sq <= tr_sq)
                    {
                        unsigned long r_root = r + 1, c_root = adjoint_table[r][c];
                        while(adjoint_table_head[r_root-1]!=r_root)r_root = adjoint_table_head[r_root-1];
                        while(adjoint_table_head[c_root-1]!=c_root)c_root = adjoint_table_head[c_root-1];

                        if(r_root < c_root)
                        {adjoint_table_head[c_root-1]=r_root;}
                        else
                        {adjoint_table_head[r_root-1]=c_root;}
                    }
                }
            }
            for(unsigned long ind = 0;ind < adjoint_table_head.size();ind++)
            {
                unsigned long ind_root = ind + 1;
                while(adjoint_table_head[ind_root-1]!=ind_root)ind_root = adjoint_table_head[ind_root-1];
                adjoint_table_head[ind] = ind_root;
            }

            Row new_Colors;
            new_Colors.reserve(region);
            for(int ind = 0;ind < region;ind++){new_Colors.emplace_back(0,0,0);}
            std::vector<int> new_pointnum;
            new_pointnum.reserve(region);
            for(int ind = 0;ind < region;ind++){new_pointnum.push_back(0);}
            for(unsigned long ind = 0;ind < adjoint_table_head.size();ind++)
            {
                unsigned long ind_ = adjoint_table_head[ind] - 1;
                new_pointnum[ind_] += pointnum[ind];
                for(int channel = 0;channel < channels;channel++)
                {
                    new_Colors[ind_][channel] += Colors[ind][channel]*pointnum[ind];
                }
            }

            std::vector<int> new_adjoint_table_head;
            new_adjoint_table_head.reserve(region);
            for(int ind = 0;ind < region;ind++){new_adjoint_table_head.push_back(0);}
            int label = 0;
            Colors.clear();
            pointnum.clear();
            for(int ind = 0;ind < region;ind++)
            {
                unsigned long ind_ = adjoint_table_head[ind] - 1;
                if(new_adjoint_table_head[ind_] <= 0)
                {
                    label++;
                    new_adjoint_table_head[ind_] = label;
                    num = new_pointnum[ind_];
                    Color cl = new_Colors[ind_];
                    pointnum.push_back(num);
                    Colors.emplace_back(cl[0]/num,cl[1]/num,cl[2]/num);
                }
            }
            std::vector<std::vector<int> > new_adjoint_table;
            new_adjoint_table.reserve(label);
            for(int ind = 0;ind < label;ind++){new_adjoint_table.emplace_back();}
            for(int ind = 0;ind < region;ind++)
            {
                int ind_ = new_adjoint_table_head[adjoint_table_head[ind] - 1] - 1;
                for(int j = 0;j < adjoint_table[ind].size();j++)
                {
                    int new_label = new_adjoint_table_head[adjoint_table_head[adjoint_table[ind][j] - 1] - 1];
                    if(new_label != ind_ + 1)
                    {
                        std::vector<int>::iterator it = std::find(new_adjoint_table[ind_].begin(),new_adjoint_table[ind_].end(),new_label);
                        if(it == new_adjoint_table[ind_].end())
                        {
                            new_adjoint_table[ind_].push_back(new_label);
                        }
                    }
                }
            }

            deltaRegion = region - label;
            region = label;
            adjoint_table = new_adjoint_table;
            for(unsigned long r = 0; r < Label.size(); r++)
            {
                for(unsigned long c = 0; c < Label[r].size(); c++)
                {
                    Label[r][c] = new_adjoint_table_head[adjoint_table_head[Label[r][c]-1]-1];
                }
            }
            adjoint_table_head.clear();
            adjoint_table_head.reserve(region);
            for(int ind = 0;ind < region;ind++){adjoint_table_head.push_back(ind+1);}
        }

        // remove small areas.
        double mindist_sq;
        for(int deltaRegion = 1;deltaRegion > 0;)
        {
            for(unsigned long r = 0;r < adjoint_table.size();r++)
            {
                mindist_sq = 1e12;  // Bigger than any possible color distance square.
                if(pointnum[r] < minarea)
                {
                    unsigned long c_root = r + 1;
                    for(unsigned long c = 0;c < adjoint_table[r].size();c++)
                    {
                        double color_distance_sq = 0;
                        Color color1 = Colors[r], color2 = Colors[adjoint_table[r][c] - 1];
                        for (int channel = 0; channel < channels; channel++)
                        {
                            double temp = color1[channel] - color2[channel];
                            color_distance_sq += (temp * temp);
                        }
                        if(color_distance_sq < mindist_sq)
                        {
                            mindist_sq = color_distance_sq;
                            c_root = adjoint_table[r][c];
                        }
                    }

                    unsigned long r_root = r + 1;
                    while(adjoint_table_head[r_root-1]!=r_root)r_root = adjoint_table_head[r_root-1];
                    while(adjoint_table_head[c_root-1]!=c_root)c_root = adjoint_table_head[c_root-1];

                    if(r_root < c_root)
                    {adjoint_table_head[c_root-1]=r_root;}
                    else
                    {adjoint_table_head[r_root-1]=c_root;}
                }
            }
            for(unsigned long ind = 0;ind < adjoint_table_head.size();ind++)
            {
                unsigned long ind_root = ind + 1;
                while(adjoint_table_head[ind_root-1]!=ind_root)ind_root = adjoint_table_head[ind_root-1];
                adjoint_table_head[ind] = ind_root;
            }

            Row new_Colors;
            new_Colors.reserve(region);
            for(int ind = 0;ind < region;ind++){new_Colors.emplace_back(0,0,0);}
            std::vector<int> new_pointnum;
            new_pointnum.reserve(region);
            for(int ind = 0;ind < region;ind++){new_pointnum.push_back(0);}
            for(unsigned long ind = 0;ind < adjoint_table_head.size();ind++)
            {
                unsigned long ind_ = adjoint_table_head[ind] - 1;
                new_pointnum[ind_] += pointnum[ind];
                for(int channel = 0;channel < channels;channel++)
                {
                    new_Colors[ind_][channel] += Colors[ind][channel]*pointnum[ind];
                }
            }

            std::vector<int> new_adjoint_table_head;
            new_adjoint_table_head.reserve(region);
            for(int ind = 0;ind < region;ind++){new_adjoint_table_head.push_back(0);}
            int label = 0;
            Colors.clear();
            pointnum.clear();
            for(int ind = 0;ind < region;ind++)
            {
                unsigned long ind_ = adjoint_table_head[ind] - 1;
                if(new_adjoint_table_head[ind_] <= 0)
                {
                    label++;
                    new_adjoint_table_head[ind_] = label;
                    num = new_pointnum[ind_];
                    Color cl = new_Colors[ind_];
                    pointnum.push_back(num);
                    Colors.emplace_back(cl[0]/num,cl[1]/num,cl[2]/num);
                }
            }
            std::vector<std::vector<int> > new_adjoint_table;
            new_adjoint_table.reserve(label);
            for(int ind = 0;ind < label;ind++){new_adjoint_table.emplace_back();}
            for(int ind = 0;ind < region;ind++)
            {
                int ind_ = new_adjoint_table_head[adjoint_table_head[ind] - 1] - 1;
                for(int j = 0;j < adjoint_table[ind].size();j++)
                {
                    int new_label = new_adjoint_table_head[adjoint_table_head[adjoint_table[ind][j] - 1] - 1];
                    if(new_label != ind_ + 1)
                    {
                        std::vector<int>::iterator it = std::find(new_adjoint_table[ind_].begin(),new_adjoint_table[ind_].end(),new_label);
                        if(it == new_adjoint_table[ind_].end())
                        {
                            new_adjoint_table[ind_].push_back(new_label);
                        }
                    }
                }
            }

            deltaRegion = region - label;
            region = label;
            adjoint_table = new_adjoint_table;
            for(unsigned long r = 0; r < Label.size(); r++)
            {
                for(unsigned long c = 0; c < Label[r].size(); c++)
                {
                    Label[r][c] = new_adjoint_table_head[adjoint_table_head[Label[r][c]-1]-1];
                }
            }
            adjoint_table_head.clear();
            adjoint_table_head.reserve(region);
            for(int ind = 0;ind < region;ind++){adjoint_table_head.push_back(ind+1);}
        }

        /////////////// build dst image ///////////////////
        dst.reserve(Label.size());
        for(unsigned long i = 0;i < Label.size();i++)
        {
            dst.emplace_back();
            dst[i].reserve(Label[i].size());
            for(unsigned long j = 0;j < Label[i].size();j++)
            {
                dst[i].push_back(Colors[Label[i][j]-1]);
            }
        }
    }

}
#endif //IMAGEPROCESSINGFROMSCRATCH_MEANSHIFT_H
