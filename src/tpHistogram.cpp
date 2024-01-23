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
    // Clone original image
    Mat res = image.clone();
    
    // Loop through each pixel and invert the value
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            float pixelValue = image.at<float>(y, x);
            res.at<float>(y, x) = 1.0 - pixelValue;
        }
    }
    return res;
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
    // Clone of the input image to store the result
    Mat res = image.clone();

    // Ensure that lowT is less than or equal to highT
    assert(lowT <= highT);

    // Iterate through each pixel of the image
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            // Get the value of the pixel at position (i, j)
            float pixelValue = image.at<float>(i, j);

            // Compare the pixel value with the thresholds
            if (pixelValue <= lowT) {
                res.at<float>(i, j) = 0.0;
            }
            else if (lowT < pixelValue && pixelValue <= highT) {
                res.at<float>(i, j) = pixelValue;
            }
            else {
                res.at<float>(i, j) = 1.0;
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
    assert(numberOfLevels > 0);

    // Loop through each pixel in the image
    for (int i = 0; i < res.rows; i++) {
        for (int j = 0; j < res.cols; j++) {
            for(int k = 0; k < numberOfLevels; k++) {

                if (res.at<float>(i, j) < (1.0 / numberOfLevels) * (k + 1)) {
                    res.at<float>(i, j) = (1.0 / (numberOfLevels - 1)) * k;
                    break;
                }
            }
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

    // Find the minimum and maximum values in the image
    double minVal, maxVal;
    minMaxLoc(res, &minVal, &maxVal);

    // Find the minimum and maximum values in the image
    for (int i = 0; i < res.rows; i++) {
        for (int j = 0; j < res.cols; j++) {
            // Normalize the pixel value within the range [minValue, maxValue]
            res.at<float>(i, j) = ((res.at<float>(i, j) - minVal) / (maxVal - minVal)) * (maxValue - minValue) + minValue;
        }
    }

    return res;
}

/**
    Equalize image histogram with unsigned char values ([0;255])

    Warning: this time, image values are unsigned chars but calculation will be done in float or double format.
    The final result must be rounded toward the nearest integer 
*/
Mat equalize(Mat image)
{
    Mat res = image.clone();
    
    // Initialize a histogram and cumulative histogram
    std::vector<int> hist(256, 0);
    std::vector<int> cumHist(256, 0);
    int totalPixels = image.rows * image.cols;

    // Calculate the histogram of the input image
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            uchar pixelValue = image.at<uchar>(i, j);
            hist[pixelValue]++;
        }
    }

    // Calculate the cumulative histogram
    cumHist[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cumHist[i] = cumHist[i - 1] + hist[i];
    }

    // Apply histogram equalization to the input image
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            uchar pixelValue = image.at<uchar>(i, j);
            float normalizedValue = static_cast<float>(cumHist[pixelValue]) / totalPixels * 255.0f;

            // Round and set the normalized value as the result pixel value
            res.at<uchar>(i, j) = cvRound(normalizedValue);
        }
    }

    return res;
}

/**
    Compute a binarization of the input float image using an automatic Otsu threshold.
    Input image is of type unsigned char ([0;255])
*/
Mat thresholdOtsu(Mat image) {
    Mat res = image.clone();

    int hist[256] = {0}; // Initialize the histogram

    // Calculate the histogram
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int intensity = image.at<uchar>(y, x);
            hist[intensity]++;
        }
    }

    int totalPixels = image.rows * image.cols;
    float sum = 0.0;

    for (int i = 0; i < 256; i++) {
        sum += i * float((hist[i]));
    }

    float sumLower = 0.0;
    int weightLower = 0;
    int weightUpper = 0;
    float maxVar = 0.0;
    int threshold = 0;

    for (int i = 0; i < 256; i++) {
        weightLower += hist[i];
        if (weightLower == 0) continue;

        weightUpper = totalPixels - weightLower;
        if (weightUpper == 0) break;

        sumLower += i * float((hist[i]));

        float meanPixelLower = sumLower / weightLower;
        float meanPixelUpper = (sum - sumLower) / weightUpper;

        float varBetween = float((weightLower)) * float((weightUpper) * (meanPixelLower - meanPixelUpper) * (meanPixelLower - meanPixelUpper));
        if (varBetween > maxVar) {
            maxVar = varBetween;
            threshold = i;
        }
    }

    // Binarize the image using the found threshold
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (image.at<uchar>(y, x) > threshold) {
                res.at<uchar>(y, x) = 255; // White
            } else {
                res.at<uchar>(y, x) = 0;   // Black
            }
        }
    }
    return res;
}
