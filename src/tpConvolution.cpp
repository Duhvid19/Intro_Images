
#include "tpConvolution.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;
/**
    Compute a mean filter of size 2k+1.

    Pixel values outside of the image domain are supposed to have a zero value.
*/
cv::Mat meanFilter(cv::Mat image, int k){
    Mat res = image.clone();
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){

            float sum = 0.0;
            for(int j = -k; j <= k ; j++){
                for(int i = -k; i <= k; i++){
                    if (y+j >= 0 && y+j < image.rows && x+i >= 0 && x+i < image.cols){
                        sum += image.at<float>(y+j, x+i);
                    }
                }
            }
            res.at<float>(y, x) = sum/((2*k+1)*(2*k+1));
        }
    }
    return res;
}

/**
    Compute the convolution of a float image by kernel.
    Result has the same size as image.
    
    Pixel values outside of the image domain are supposed to have a zero value.
*/
Mat convolution(Mat image, cv::Mat kernel)
{
    Mat res = image.clone();
    int border = (kernel.cols-1)/2;
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){

            float sum = 0.0;
            for(int j = -border; j <= border ; j++){
                for(int i = -border; i <= border; i++){
                    if (y+j >= 0 && y+j < image.rows && x+i >= 0 && x+i < image.cols){ // si pixel hors de l'image -> il vaut 0 pour la convolution
                        sum += image.at<float>(y+j, x+i) * kernel.at<float>(j+border, i+border);
                    }
                }
            }
            res.at<float>(y, x) = sum;
        }
    }
    return res;
}

/**
    Compute the sum of absolute partial derivative according to Sobel's method
*/
cv::Mat edgeSobel(cv::Mat image)
{
    float x_data[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    cv::Mat kernel_x(3, 3, CV_32F, x_data);

    float y_data[] = {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0};
    cv::Mat kernel_y(3, 3, CV_32F, y_data);

    return cv::abs(convolution(image, kernel_x)) + cv::abs(convolution(image, kernel_y));
}

/**
    Value of a centered gaussian of variance (scale) sigma at point x.
*/
float gaussian(float x, float sigma2)
{
    return 1.0/(2*M_PI*sigma2)*exp(-x*x/(2*sigma2));
}

/**
    Performs a bilateral filter with the given spatial smoothing kernel 
    and a intensity smoothing of scale sigma_r.

*/
cv::Mat bilateralFilter(cv::Mat image, cv::Mat kernel, float sigma_r)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
   
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
