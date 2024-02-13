# image processing from scratch

## what is this?
This repository contains many interesting image processing algorithms that are written from scratch. Read these codes will allow you to have a comprehensive understanding of the principles of these algorithms. 

<strong>Implementation</strong>  
All codes were wrote in _python3.7_ or _c++_  
moudles you may need:  
_python:_ 
- numpy for matix calculation  
- matplotlib for reading and showing images  
- opencv2 for some image operations  

_c++:_  
- opencv2  

<strong>Usage</strong>  
you can always run a python script just by  
    `python script.py`  

for c++, you need to compile first  
    `cd build`  
    `cmake ..`  
    `make`  

when it's done, you are ready to run the executable file by  
    `./program_name parameters`  
Just make sure you have the images in the right path, and you might wanna modify the code a bit to process another image.  
<strong>Have fun!</strong>  

## Contents
* canny edge detection  
It is an algorithm that extracts edges of an image.  

* hough transform  
It is an algorithm that can theoratically detects shapes that you can write formulas for it.  

* harris corner detection  
This algorithm detects corners.  

* fast fourier transform  
2-D fourier transform for images using fft.  

* sift  
Scale-invariant feature transform, a well-known technique to extract feature points for image matching. Now added c++ version along with SURF and ORB.  

* KNN  
Using balanced K-D tree to find k nearest neighbors of K-dimension points.  

* PCA&SVD  
Do PCA and SVD using jacobi rotation.(which is accurate but slow)  

* Ransac  
Stitch different images together after knowing the sift keypoint pairs.  

* watershed  
watershed segmentation algorithm.  

* meanshift  
meanshift segmentation algorithm.  

* generalized hough transform  
template match of images, detects a given template in an query image. The vote space is implemented with a sparse vector to support big images.  

* closed-form image matting  
  a classic image matting algorithm proposed in ***A Closed-Form Solution to Natural Image Matting***  
  
* haze removal  
  Using dark channel prior and fast guided filter proposed in ***Single Image Haze Removal Using Dark Channel Prior*** and ***Fast Guided Filter***
  
* a lot to be continued...

