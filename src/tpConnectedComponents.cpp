#include "tpConnectedComponents.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <map>
#include <stack>
using namespace cv;
using namespace std;


/**
    Performs a labeling of image connected component with 4 connectivity
    with a depth-first exploration.
    Any non zero pixel of the image is considered as present.
*/
void parcoursCC(Mat& image, Mat& res, int label, const vector<Point>& voisins, const Point& start) {
    stack<Point> s;
    s.push(start);

    while (!s.empty()) {
        Point r = s.top();
        s.pop();
        res.at<int>(r) = label;

        for (Point voisin : voisins) {
            Point v = r + voisin;
            if (v.x >= 0 && v.x < image.cols && v.y >= 0 && v.y < image.rows &&
                image.at<float>(v) != 0 && res.at<int>(v) == 0) {
                s.push(v);
            }
        }
    }
}

Mat ccLabel(Mat image) {
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1);
    int label = 1;

    vector<Point> voisins = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (image.at<float>(y, x) != 0 && res.at<int>(y, x) == 0) {
                parcoursCC(image, res, label, voisins, Point(x, y));
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

    map<int, int> compSizes;

    int y = 0;
    while (y < labels.rows) {
        int x = 0;
        while (x < labels.cols) {
            int label = labels.at<int>(y, x);
            if (label > 0) {
                compSizes[label]++;
            }
            x++;
        }
        y++;
    }

    y = 0;
    while (y < labels.rows) {
        int x = 0;
        while (x < labels.cols) {
            int label = labels.at<int>(y, x);
            if (label > 0 && compSizes[label] >= size) {
                res.at<float>(y, x) = image.at<float>(y, x);
            }
            x++;
        }
        y++;
    }
    return res;
}

/**
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/

cv::Mat ccLabel2pass(cv::Mat image)
{
    int numRows = image.rows;
    int numCols = image.cols;
    int labelCounter = 0;
    std::vector<std::vector<int>> labels(numRows, std::vector<int>(numCols, 0));
    std::map<int, int> labelEquivalences;  // Déclaration de labelsEquivalence ici

    for (int y = 0; y < numRows; y++) {
        for (int x = 0; x < numCols; x++) {
            if (image.at<float>(y, x) != 0) {
                std::vector<int> neighboringLabels;

                if (x > 0 && labels[y][x - 1] != 0)
                    neighboringLabels.push_back(labels[y][x - 1]);
                if (y > 0 && labels[y - 1][x] != 0)
                    neighboringLabels.push_back(labels[y - 1][x]);

                if (neighboringLabels.empty()) {
                    labelCounter++;
                    labels[y][x] = labelCounter;
                } else {
                    labels[y][x] = *std::min_element(neighboringLabels.begin(), neighboringLabels.end());
                    for (int label : neighboringLabels) {
                        if (label != labels[y][x]) {
                            // Mettre à jour les équivalences de labels
                            int currentLabel = labels[y][x];
                            int neighborLabel = label;

                            while (labelEquivalences.find(currentLabel) != labelEquivalences.end()) {
                                currentLabel = labelEquivalences[currentLabel];
                            }

                            while (labelEquivalences.find(neighborLabel) != labelEquivalences.end()) {
                                neighborLabel = labelEquivalences[neighborLabel];
                            }

                            if (currentLabel != neighborLabel) {
                                labelEquivalences[currentLabel] = neighborLabel;
                            }
                        }
                    }
                }
            }
        }
    }

    // Réaffecter les labels
    std::map<int, int> newLabels;
    int newLabel = 1;
    for (int y = 0; y < numRows; y++) {
        for (int x = 0; x < numCols; x++) {
            if (labels[y][x] != 0) {
                int rootLabel = labels[y][x];

                while (labelEquivalences.find(rootLabel) != labelEquivalences.end()) {
                    rootLabel = labelEquivalences[rootLabel];
                }

                if (newLabels.find(rootLabel) == newLabels.end()) {
                    newLabels[rootLabel] = newLabel;
                    newLabel++;
                }

                labels[y][x] = newLabels[rootLabel];
            }
        }
    }

    cv::Mat result(numRows, numCols, CV_32SC1);
    for (int y = 0; y < numRows; y++) {
        for (int x = 0; x < numCols; x++) {
            result.at<int>(y, x) = labels[y][x];
        }
    }

    return result;
}