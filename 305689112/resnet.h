#ifndef RESNET_H
#define RESNET_H

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

class ResNetFeatures {
public:
    ResNetFeatures(const std::string& model_path);
    std::vector<float> getFeatures(const std::string& filename);  // From file
    std::vector<float> getFeatures(const cv::Mat& img);          // From Mat
    float computeDistance(const std::vector<float>& feat1, const std::vector<float>& feat2);

private:
    std::string model_path;
    const int FEATURE_DIM = 512;
};

#endif
