#ifndef SPLIT_RGB_HIST_H
#define SPLIT_RGB_HIST_H

#include <vector>
#include "opencv2/opencv.hpp"

std::vector<float> computeSplitRGBHist(const cv::Mat& img);
float computeHistIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2);

#endif
