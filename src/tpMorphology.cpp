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
    Mat res(image.size(), CV_32FC1);

    /********************************************
                YOUR CODE HERE
    *********************************************/

    int i = 0;
    while (i < image.rows) 
    {
        int j = 0;
        while (j < image.cols) 
        {
            std::vector<float> pixVoisin;
            int minLigne = std::max(i - size, 0);
            int maxLigne = std::min(i + size, image.rows - 1);
            int minColonne = std::max(j - size, 0);
            int maxColonne = std::min(j + size, image.cols - 1);

            int x = minLigne;
            while (x <= maxLigne) 
            {
                int y = minColonne;
                while (y <= maxColonne)    
                {
                    pixVoisin.push_back(image.at<float>(x, y));
                    y++;
                }
                x++;
            }

            // Tri manuel des valeurs
            int n = pixVoisin.size();
            int m = 0;
            while (m < n - 1) 
            {
                int k = 0;
                while (k < n - m - 1) 
                {
                    switch (pixVoisin[k] > pixVoisin[k + 1]) 
                    {
                        case true:
                            float temp = pixVoisin[k];
                            pixVoisin[k] = pixVoisin[k + 1];
                            pixVoisin[k + 1] = temp;
                            break;
                    }
                    k++;
                }
                m++;
            }

            float mediane = 0;
            if (n % 2 == 0) 
            {
                mediane = (pixVoisin[n / 2 - 1] + pixVoisin[n / 2]) / 2;
            }
            else 
            {
                mediane = pixVoisin[n / 2];
            }

            res.at<float>(i, j) = mediane;
            j++;
        }
        i++;
    }
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
    Mat res(image.size(), CV_32FC1);

    int largeurElementStructur =structuringElement.rows/2;
    int hauteurElementStructur =structuringElement.cols/2;

    // parcours chaque elem du res pixel
    res.forEach<float>([&](float& pix, const int* pos) {
        int debut = pos[0];
        int fin= pos[1];

        // initialisation de vecteur pour stocker les val des pix nearbor
        std::vector<float> nbPix ={};

        int premier =-largeurElementStructur;
        while (premier <= largeurElementStructur) {
            int deuxieme= -hauteurElementStructur;
            while (deuxieme<= hauteurElementStructur) {
                int xDehors = premier + largeurElementStructur;
                int yDehors = deuxieme+ hauteurElementStructur;

                if (debut+premier >= 0 && debut + premier < image.rows && fin+ deuxieme>= 0 && fin+ deuxieme< image.cols) {
                    float valStruc = structuringElement.at<float>(xDehors, yDehors);
                    
                    if (valStruc==1) {
                        nbPix.push_back(image.at<float>(debut + premier, fin+deuxieme));
                    }
                }
                deuxieme++;
            }
            premier++;
        }

        if (nbPix.size()>0) {
            float max = nbPix[0];
            size_t trie = 1;
            while (trie<nbPix.size()) {
                if (nbPix[trie] > max) {
                    max = nbPix[trie];
                }
                trie++;
            }
        pix = max;
        }
    });

    return res;
}


/**
    Compute the erosion of the input float image by the given structuring element.
    Pixel outside the image are supposed to have value 1.
*/
Mat erode(Mat image, Mat structuringElement)
{
    Mat res = image.clone();

    int largeurElementStructur = (structuringElement.rows - 1) / 2;
    int hauteurElementStructur = (structuringElement.cols - 1) / 2;

    int ligne = 0;
    while (ligne < image.rows)
    {
        int colonne = 0;
        while (colonne < image.cols)
        {
            float valeurMinimumPixel = 1.0;

            int x = -largeurElementStructur;
            while (x <= largeurElementStructur)
            {
                int y = -hauteurElementStructur;
                while (y <= hauteurElementStructur)
                {
                    if (structuringElement.at<float>(x + largeurElementStructur, y + hauteurElementStructur) == 1)
                    {
                        int ligneImage = ligne + x;
                        int colonneImage = colonne + y;

                        if (ligneImage >= 0 && ligneImage < image.rows && colonneImage >= 0 && colonneImage < image.cols)
                        {
                            float valeurPixel = image.at<float>(ligneImage, colonneImage);
                            valeurMinimumPixel = std::min(valeurMinimumPixel, valeurPixel);
                        }
                    }
                    y++;
                }
                x++;
            }

            res.at<float>(ligne, colonne) = valeurMinimumPixel;

            colonne++;
        }
        ligne++;
    }

    return res;
}


/**
    Compute the opening of the input float image by the given structuring element.
*/
Mat open(Mat image, Mat structuringElement)
{

    // Mat res = Mat::zeros(1,1,CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/
    Mat erosionIm = erode(image, structuringElement);
    Mat res = dilate(erosionIm, structuringElement);
    return res;
}


/**
    Compute the closing of the input float image by the given structuring element.
*/
Mat close(Mat image, Mat structuringElement)
{

    // Mat res = Mat::zeros(1,1,CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/
    Mat dilatationIm = dilate(image, structuringElement);
    Mat res = erode(dilatationIm, structuringElement);
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}


/**
    Compute the morphological gradient of the input float image by the given structuring element.
*/
Mat morphologicalGradient(Mat image, Mat structuringElement)
{

    Mat res = Mat::zeros(1,1,CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/
    Mat dilatationIm = dilate(image, structuringElement);
    Mat erosionIm = erode(image, structuringElement);
    res = dilatationIm - erosionIm;
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
