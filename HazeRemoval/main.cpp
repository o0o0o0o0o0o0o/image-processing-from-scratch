//
// author: 会飞的吴克
//
#include "dehaze.h"

int main(int argc,char** argv)
{
    __Assert__(argc == 2,"Please provide a correct image name.");
    std::string img_name = argv[1];
    int pos = img_name.find_last_of('/');
    std::string path;
    if(pos == std::string::npos){path = "";}
    else{path = img_name.substr(0,pos+1);img_name = img_name.substr(pos+1);}
    cv::Mat img = cv::imread(path+img_name);
    cv::Mat res = DarkChannel::dehaze(img);
    std::cout << res.rows << " " << res.cols << std::endl;
    cv::imwrite("dehazed_"+img_name,res);
    cv::imshow("result",res);
    cv::waitKey(0);
    return 0;
}