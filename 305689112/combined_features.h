#ifndef COMBINED_FEATURES_H
#define COMBINED_FEATURES_H

#include <vector>
#include "opencv2/opencv.hpp"
#include "resnet.h"
#include "color_texture_hist.h"
#include "split_rgb_hist.h"

class CombinedFeatures {
public:
    CombinedFeatures();
    std::vector<float> computeFeatures(const cv::Mat& img);
    float computeDistance(const std::vector<float>& feat1, const std::vector<float>& feat2);

private:
    ResNetFeatures resnet;
    std::vector<float> computeResNetFeatures(const cv::Mat& img);
    std::vector<float> computeForegroundFeatures(const cv::Mat& img);
    std::vector<float> computeCentralRGBHist(const cv::Mat& img);
    
    const float RESNET_WEIGHT = 0.4f;
    const float FOREGROUND_WEIGHT = 0.4f;
    const float CENTER_WEIGHT = 0.2f;
};

#endif
