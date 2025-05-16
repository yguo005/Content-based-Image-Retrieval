#ifndef COLOR_TEXTURE_HIST_H
#define COLOR_TEXTURE_HIST_H

#include <vector>
#include "opencv2/opencv.hpp"

std::vector<float> computeColorTextureHist(const cv::Mat& img);
float computeCombinedDistance(const std::vector<float>& hist1, 
                            const std::vector<float>& hist2);

#endif
