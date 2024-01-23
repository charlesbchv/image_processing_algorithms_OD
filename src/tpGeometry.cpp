#include "tpGeometry.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

/**
    Transpose the input image,
    ie. performs a planar symmetry according to the
    first diagonal (upper left to lower right corner).
*/
Mat transpose(Mat image)
{
    /********************************************
                YOUR CODE HERE
    *********************************************/
    Mat res(image.cols, image.rows, CV_32FC1);

    int y = 0;
    while (y < image.rows)
    {
        int x =0;
        while (x < image.cols)
        {
            res.at<float>(x, y) = image.at<float>(y, x);
            x++;
        }
        y++;
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the value of a nearest neighbour interpolation
    in image Mat at position (x,y)
*/
float interpolate_nearest(Mat image, float y, float x)
{
    int lePlusProcheK = lroundf(x);
    int lePlusProcheL = lroundf(y);
    
    return image.at<float>(lePlusProcheL, lePlusProcheK);
}

/**
    Compute the value of a bilinear interpolation in image Mat at position (x,y)
*/
float interpolate_bilinear(Mat image, float y, float x)
{
    int x1 = floor(x);
    int x2 = ceil(x);
    int y1 = floor(y);
    int y2 = ceil(y);

    float alpha = x - x1;
    float beta = y - y1;

    x1 = max(0, min(x1, image.cols - 1));
    x2 = max(0, min(x2, image.cols - 1));
    y1 = max(0, min(y1, image.rows - 1));
    y2 = max(0, min(y2, image.rows - 1));

    float f_x1y1 = image.at<float>(y1, x1);
    float f_x2y1 = image.at<float>(y1, x2);
    float f_x1y2 = image.at<float>(y2, x1);
    float f_x2y2 = image.at<float>(y2, x2);

    float interpolated_value = (1 - alpha) * (1 - beta) * f_x1y1 +
                               alpha * (1 - beta) * f_x2y1 +
                               (1 - alpha) * beta * f_x1y2 +
                               alpha * beta * f_x2y2;

    return interpolated_value;
}

/**
    Multiply the image resolution by a given factor using the given interpolation method.
    If the input size is (h,w) the output size shall be ((h-1)*factor, (w-1)*factor)
*/
Mat expand(Mat image, int factor, float(* interpolationFunction)(cv::Mat image, float y, float x))
{
    /********************************************
                YOUR CODE HERE
    *********************************************/

    assert(factor>0);
    Mat res = Mat::zeros((image.rows-1)*factor,(image.cols-1)*factor,CV_32FC1);
    int nouvelleHauteur = (image.rows - 1) * factor;
    int nouvelleLargeur = (image.cols - 1) * factor;

    int y = 0;
    while (y < nouvelleHauteur)
    {
        int x = 0;
        while (x < nouvelleLargeur)
        {
            float originalX = (float)(x) / factor;
            float originalY = (float)(y) / factor;
            res.at<float>(y, x) = interpolationFunction(image, originalY, originalX);
            x++;
        }
        y++;
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/

    return res;
}

/**
    Performs a rotation of the input image with the given angle (clockwise) and the given interpolation method.
    The center of rotation is the center of the image.

    Ouput size depends of the input image size and the rotation angle.

    Output pixels that map outside the input image are set to 0.
*/
Mat rotate(Mat image, float angle, float(*interpolationFunction)(cv::Mat image, float y, float x))
{
    /********************************************
                YOUR CODE HERE
    hint: to determine the size of the output, take
    the bounding box of the rotated corners of the 
    input image.
    *********************************************/

    // Mat res = Mat::zeros(1,1,CV_32FC1);
    float radius = angle * CV_PI/180.0;
    float xOrigine = (float)(image.cols - 1) / 2.0;
    float yOrigine = (float)(image.rows - 1) / 2.0;
    float a = -xOrigine;
    float b = image.cols - xOrigine;
    float d = -yOrigine;
    float c = image.rows - yOrigine;

    float xancien[4], yancien[4];

    for (int i = 0; i < 4; i++)
    {
        float sup = 0, inf = 0;
        switch (i)
        {
        case 0:
            sup = a * cos(radius) - d * sin(radius);
            inf = a * sin(radius) + d * cos(radius);
            break;
        case 1:
            sup = b * cos(radius) - d * sin(radius);
            inf = b * sin(radius) + d * cos(radius);
            break;
        case 2:
            sup = a * cos(radius) - c * sin(radius);
            inf = a * sin(radius) + c * cos(radius);
            break;
        case 3:
            sup = b * cos(radius) - c * sin(radius);
            inf = b * sin(radius) + c * cos(radius);
            break;
        }
        xancien[i] = sup;
        yancien[i] = inf;
    }

    int xinferieur = (int)(std::min(std::min(std::min(xancien[0], xancien[1]), xancien[2]), xancien[3]));
    int xsuperieur = (int)(std::max(std::max(std::max(xancien[0], xancien[1]), xancien[2]), xancien[3]));
    int yinferieur = (int)(std::min(std::min(std::min(yancien[0], yancien[1]), yancien[2]), yancien[3]));
    int ysuperieur = (int)(std::max(std::max(std::max(yancien[0], yancien[1]), yancien[2]), yancien[3]));
    int nouvelleLargeur = xsuperieur - xinferieur;
    int nouvelleHauteur = ysuperieur - yinferieur;

    Mat res = Mat::zeros(nouvelleHauteur, nouvelleLargeur, CV_32FC1);

    int k = 0;
    while (k < nouvelleLargeur)
    {
        int m = 0;
        while (m < nouvelleHauteur)
        {
            float x = k - ((nouvelleLargeur - 1) / 2.0);
            float y = m - ((nouvelleHauteur - 1) / 2.0);
            float sup = x * cos(-radius) - y * sin(-radius) + xOrigine;
            float inf = x * sin(-radius) + y * cos(-radius) + yOrigine;

            if (sup >= 0 && sup < image.cols - 1 && inf >= 0 && inf < image.rows - 1)
            {
                res.at<float>(m, k) = interpolationFunction(image, inf, sup);
            } else {
                res.at<float>(m, k) = 0;
            }
            m++;
        }
        k++;
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
