/*
Yunyu Guo
CS5330 Project 2
Feb 4 2025

ResNet Feature Extraction
*/

#include "resnet.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include "opencv2/opencv.hpp"

ResNetFeatures::ResNetFeatures(const std::string& path) : model_path(path) {}

std::vector<float> ResNetFeatures::getFeatures(const std::string& filename) {
    cv::Mat img = cv::imread(filename);
    return getFeatures(img);
}

std::vector<float> ResNetFeatures::getFeatures(const cv::Mat& img) {
    // Your ResNet feature extraction code here
    // For now, return a placeholder vector
    return std::vector<float>(512, 0.0f);  // 512-dimensional feature vector
}

float ResNetFeatures::computeDistance(const std::vector<float>& feat1, 
                                    const std::vector<float>& feat2) {
    // Compute cosine distance
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < feat1.size(); i++) {
        dot_product += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 > 0 && norm2 > 0) {
        return 1.0f - (dot_product / (norm1 * norm2));
    }
    return 1.0f;
}
