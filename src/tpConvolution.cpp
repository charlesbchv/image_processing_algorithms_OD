
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
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int debut= 0;
    int taille = (2 * k + 1);
    int taille_fen_carre = taille * taille;
    while (debut< image.rows) {
        int fin= 0;
        while (fin< image.cols) {
            int ligne_min = (debut- k) < 0 ? 0 : (debut- k);
            int ligne_max = (debut+ k) > (image.rows - 1) ? (image.rows - 1) : (debut+ k);
            int colonne_min = (fin- k) < 0 ? 0 : (fin- k);
            int colonne_max = (fin+ k) > (image.cols - 1) ? (image.cols - 1) : (fin+ k);
            float sommePix = 0;
            int taille_fen_actuelle = 0;

            int x = ligne_min;
            while (x <= ligne_max) {
                int y = colonne_min;
                while (y <= colonne_max) {
                    sommePix += image.at<float>(x, y);
                    taille_fen_actuelle++;
                    y++;
                }
                x++;
            }
            float moyenne = sommePix /(float)(taille_fen_carre);
            res.at<float>(debut, fin) = moyenne;
            fin+=1;
        }
        debut+=1;
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/

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
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int noeud= kernel.rows/2;
    int debut = 0;
    while (debut< image.rows) {
        int fin= 0;
        while (fin< image.cols) {
            int x= -noeud;
            float sommePix=0;

            while (x<= noeud) {
                int y= -noeud;
                while (y<= noeud) {
                    bool coordonneesValides = (debut + x >= 0 && fin+ y >= 0 && debut + x < image.rows && fin+ y < image.cols);
                    if (coordonneesValides) {
                        float pixelIm = image.at<float>(debut+ x, fin+ y);
                        float pixelNoy = kernel.at<float>(x + noeud, y + noeud);
                        sommePix += pixelIm * pixelNoy;
                    }
                    y+=1;
                }
                x+=1;
            }
            res.at<float>(debut, fin) = sommePix;
            fin++;
        }
        debut+=1;
    }

    return res;
    /********************************************
                END OF YOUR CODE
    *********************************************/

    return res;
}

/**
    Compute the sum of absolute partial derivative according to Sobel's method
*/
cv::Mat edgeSobel(cv::Mat image)
{
    cv::Mat res = cv::Mat::zeros(image.size(), CV_32F);

    cv::Mat Gx = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gy = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float xGradient = 0.0;
            float yGradient = 0.0;

            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    int newX = i + x;
                    int newY = j + y;

                    if (newX >= 0 && newX < image.rows && newY >= 0 && newY < image.cols) {
                        xGradient += image.at<float>(newX, newY) * Gx.at<float>(x + 1, y + 1);
                        yGradient += image.at<float>(newX, newY) * Gy.at<float>(x + 1, y + 1);
                    }
                }
            }

            float gradientMagnitude = std::abs(xGradient) + std::abs(yGradient);
            res.at<float>(i, j) = gradientMagnitude;
        }
    }

    return res;
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
    cv::Mat result = cv::Mat::zeros(image.size(), image.type());

    int kernelRadiusX = kernel.cols / 2;
    int kernelRadiusY = kernel.rows / 2;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            float sum = 0.0;
            float normalization = 0.0;

            for (int ky = -kernelRadiusY; ky <= kernelRadiusY; ky++) {
                for (int kx = -kernelRadiusX; kx <= kernelRadiusX; kx++) {
                    int srcX = x + kx;
                    int srcY = y + ky;

                    if (srcX >= 0 && srcX < image.cols && srcY >= 0 && srcY < image.rows) {
                        float diffIntensity = image.at<float>(y, x) - image.at<float>(srcY, srcX);
                        float weight = gaussian(diffIntensity, sigma_r * sigma_r) * kernel.at<float>(ky + kernelRadiusY, kx + kernelRadiusX);
                        sum += image.at<float>(srcY, srcX) * weight;
                        normalization += weight;
                    }
                }
            }

            result.at<float>(y, x) = sum / normalization;
        }
    }

    return result;
}