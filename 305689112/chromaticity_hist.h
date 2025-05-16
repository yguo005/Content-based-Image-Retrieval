#ifndef CHROMATICITY_HIST_H
#define CHROMATICITY_HIST_H

#include <vector>
#include "opencv2/opencv.hpp"

// Compute 16x16 rg chromaticity histogram from image
std::vector<float> computeChromaticityHist(const cv::Mat& img);

// Compute histogram intersection between two normalized histograms
float computeHistIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2);

#endif
