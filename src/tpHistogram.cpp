#include "tpHistogram.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

/**
    Inverse a grayscale image with float values.
    for all pixel p: res(p) = 1.0 - image(p)
*/
Mat inverse(Mat image)
{
    // clone original image
    Mat res = image.clone();
    return res = 1.0 - image;
}

/**
    Thresholds a grayscale image with float values.
    for all pixel p: res(p) =
        | 0 if image(p) <= lowT
        | image(p) if lowT < image(p) <= hightT
        | 1 otherwise
*/
Mat threshold(Mat image, float lowT, float highT)
{
    Mat res = image.clone();
    assert(lowT <= highT);
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            if(image.at<float>(y, x) <= lowT){
                res.at<float>(y, x) = 0.0;
            }
            else if(lowT < res.at<float>(y, x) &&  res.at<float>(y, x) <= highT)
            {
                res.at<float>(y, x) = image.at<float>(y, x);
            }
            else{
                res.at<float>(y, x) = 1.0;
            }
        }
    }
    return res;
}

/**
    Quantize the input float image in [0,1] in numberOfLevels different gray levels.
    
    eg. for numberOfLevels = 3 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/3
        | 1/2 if 1/3 <= image(p) < 2/3
        | 1 otherwise

        for numberOfLevels = 4 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/4
        | 1/3 if 1/4 <= image(p) < 1/2
        | 2/3 if 1/2 <= image(p) < 3/4
        | 1 otherwise

        and so on for other values of numberOfLevels.

*/
Mat quantize(Mat image, int numberOfLevels)
{
    Mat res = image.clone();
    assert(numberOfLevels>0);
    for(int y = 0; y < res.rows; y++){
        for(int x = 0; x < res.cols; x++){
            int level = (int)(image.at<float>(y, x) * numberOfLevels);
            if(level >= numberOfLevels){
                level = numberOfLevels - 1;
            }
            res.at<float>(y, x) = (float)level/(numberOfLevels - 1);
        }
    }
    return res;
}

/**
    Normalize a grayscale image with float values
    Target range is [minValue, maxValue].
*/
Mat normalize(Mat image, float minValue, float maxValue)
{
    Mat res = image.clone();
    assert(minValue <= maxValue);
    double min, max;
    cv::minMaxLoc(res, &min, &max);
    return (res - min) * (maxValue - minValue)/(max - min) + minValue;;
}



/**
    Equalize image histogram with unsigned char values ([0;255])

    Warning: this time, image values are unsigned chars but calculation will be done in float or double format.
    The final result must be rounded toward the nearest integer 
*/
Mat equalize(Mat image)
{
    Mat res = image.clone();

    int hist[256] = {0};
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            hist[image.at<uchar>(y, x)]++;
        }
    }
    
    float hist_cumule[256] = {0};
    hist_cumule[0] = (float)hist[0]/(image.rows * image.cols);
    for(int i = 1; i < 256; i++){
        hist_cumule[i] = hist_cumule[i-1] + (float)hist[i]/(image.rows * image.cols);
    }

    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            res.at<uchar>(y, x) = round(hist_cumule[image.at<uchar>(y, x)] * 255);
        }
    }
    return res;
}

/**
    Compute a binarization of the input float image using an automatic Otsu threshold.
    Input image is of type unsigned char ([0;255])
*/
Mat thresholdOtsu(Mat image)
{
    Mat res = image.clone();

    int hist[256] = {0};
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            hist[image.at<uchar>(y, x)]++;
        }
    }

    float hist_norm[256] = {0};
    for(int i = 0; i < 256; i++){
        hist_norm[i] = (float)hist[i]/(image.rows * image.cols);
    }

    float sum = 0.0;
    for(int i = 0; i < 256; i++){
        sum += i * hist_norm[i];
    }

    float sum1 = 0.0;
    float w1 = 0.0;
    float w2 = 0.0;
    float max_var = 0.0;
    int threshold = 0;

    for(int i = 0; i < 256; i++){
        w1 += hist_norm[i];
        if(w1 == 0.0){
            continue;
        }
        w2 = 1.0 - w1;
        if(w2 == 0.0){
            break;
        }
        sum1 += i * hist_norm[i];

        float m1 = sum1/w1;
        float m2 = (sum - sum1)/w2;

        float var = w1 * w2 * (m1 - m2) * (m1 - m2);

        if(var > max_var){
            max_var = var;
            threshold = i;
        }
    }

    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            if(image.at<uchar>(y, x) > threshold){
                res.at<uchar>(y, x) = 255;
            } 
            else{
                res.at<uchar>(y, x) = 0;
            }
        }
    }

    return res;
}