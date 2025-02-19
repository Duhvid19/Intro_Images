#include "tpMorphology.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <limits>
#include "common.h"
using namespace cv;
using namespace std;


/**
    Compute a median filter of the input float image.
    The filter window is a square of (2*size+1)*(2*size+1) pixels.

    Values outside image domain are ignored.

    The median of a list l of n>2 elements is defined as:
     - l[n/2] if n is odd 
     - (l[n/2-1]+l[n/2])/2 is n is even 
*/
Mat median(Mat image, int size)
{
    Mat res = image.clone();
    assert(size>0);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}


/**
    Compute the dilation of the input float image by the given structuring element.
     Pixel outside the image are supposed to have value 0
*/
Mat dilate(Mat image, Mat structuringElement)
{
    Mat res = image.clone();
    int border_y = (structuringElement.rows-1)/2;
    int border_x = (structuringElement.cols-1)/2;
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){

            float max_value = 0.0;
            for(int j = -border_y; j <= border_y ; j++){
                for(int i = -border_x; i <= border_x; i++){
                    if (structuringElement.at<float>(j + border_y, i + border_x) != 0){
                        if (y+j >= 0 && y+j < image.rows && x+i >= 0 && x+i < image.cols){ // si pixel hors de l'image -> il vaut 0 pour la convolution
                            float value = image.at<float>(y+j, x+i);
                            if(max_value < value){
                                max_value = value;
                            }
                        }
                    }
                }
            }
            res.at<float>(y, x) = max_value;
        }
    }
    return res;
}


/**
    Compute the erosion of the input float image by the given structuring element.
    Pixel outside the image are supposed to have value 1.
*/
Mat erode(Mat image, Mat structuringElement)
{
    return 1.0-dilate(1.0-image, structuringElement);
}


/**
    Compute the opening of the input float image by the given structuring element.
*/
Mat open(Mat image, Mat structuringElement)
{
    return 1.0-close(1.0-image, structuringElement);
}


/**
    Compute the closing of the input float image by the given structuring element.
*/
Mat close(Mat image, Mat structuringElement)
{
    return erode(dilate(image, structuringElement), structuringElement);
}


/**
    Compute the morphological gradient of the input float image by the given structuring element.
*/
Mat morphologicalGradient(Mat image, Mat structuringElement)
{
    return dilate(image, structuringElement) - erode(image, structuringElement);
}

