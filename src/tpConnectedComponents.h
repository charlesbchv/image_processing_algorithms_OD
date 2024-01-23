
#pragma once

#include <opencv2/opencv.hpp>

cv::Mat ccLabel(cv::Mat image);

cv::Mat ccAreaFilter(cv::Mat image, int size);

cv::Mat ccLabel2pass(cv::Mat image);