#include "tpConnectedComponents.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <map>
using namespace cv;
using namespace std;


/**
    Performs a labeling of image connected component with 4 connectivity
    with a depth-first exploration.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    int label = 1;

    vector<Point2i> neighbours = {{-1,0}, {0,-1}, {0,1}, {1,0}};

    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            if(image.at<int>(y, x) != 0.0 && res.at<int>(y, x) == 0){ // la matrice res permet egalement de connaiatre les points visites

                vector<Point2i> s; // c'est la pile s
                s.push_back(Point2i(x, y));
                res.at<int>(y, x) = label;

                while(!s.empty()){
                    Point2i r = s.back();
                    s.pop_back();

                    for(Point2i v: neighbours){
                        v += r;
                        if (v.x >= 0 && v.x < image.cols && v.y >= 0 && v.y < image.rows){ // si le voisin v reste bien dans l'image
                            if (image.at<int>(v.y, v.x) != 0 && res.at<int>(v.y, v.x) == 0){
                                res.at<int>(v.y, v.x) = label; 
                                s.push_back(v);
                            }
                        }
                    }
                }
                label++;
            }
        }
    }
    return res;
}

/**
    Deletes the connected components (4 connectivity) containg less than size pixels.
*/
cv::Mat ccAreaFilter(cv::Mat image, int size)
{
    Mat res = Mat::zeros(image.rows, image.cols, image.type());
    assert(size>0);
    Mat labels = ccLabel(image);

    map<int, int> dict; // dictionnaire  (cle = label, valeur = taille) 
    for (int y = 0; y < labels.rows; y++){
        for (int x = 0; x < labels.cols; x++){
            int label = labels.at<int>(y, x);
            if (label != 0){
                dict[label]++;
            }
        }
    }

    for (int y = 0; y < labels.rows; y++){
        for (int x = 0; x < labels.cols; x++){
            int label = labels.at<int>(y, x);
            if (label != 0 && dict[label] >= size){ // si la composante contient assez de pixels, le pixel reste
                res.at<int>(y, x) = image.at<int>(y, x);
            }
        }
    }
    return res;
}


/**
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccTwoPassLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32-bit integer image

    // pas reussi ;-;

    return res;
}