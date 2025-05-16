/*
Yunyu Guo
CS5330 Project 2
Jan 28 2025
*/

#ifndef FEATURE_UTIL_H
#define FEATURE_UTIL_H

#include <vector>
#include "opencv2/opencv.hpp"

//share this function between files
std::vector<float> extractCenterPatch(const cv::Mat& img);

#endif
