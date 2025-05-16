/*
Yunyu Guo
CS5330 Project 2
Feb 4 2025

ResNet18 features for semantic understanding
Color-texture histogram on foreground
Split RGB histogram on central region
Weighted distance combination: implement in header file
ResNet18: 40%
Foreground: 40%
Central: 20%
 */

#include "combined_features.h"
#include "color_texture_hist.h"
#include "split_rgb_hist.h"
#include <numeric> // for std::accumulate

CombinedFeatures::CombinedFeatures() : resnet("../ResNet18_olym.csv") {}

std::vector<float> CombinedFeatures::computeFeatures(const cv::Mat& img) {
    std::vector<float> combined_features;
    
    // 1. Get ResNet features (40%)
    std::vector<float> resnet_features = computeResNetFeatures(img);
    
    // 2. Get foreground color-texture features (40%)
    std::vector<float> foreground_features = computeForegroundFeatures(img);
    
    // 3. Get central RGB histogram features (20%)
    std::vector<float> central_features = computeCentralRGBHist(img);
    
    // Combine all features
    combined_features.insert(combined_features.end(), resnet_features.begin(), resnet_features.end());
    combined_features.insert(combined_features.end(), foreground_features.begin(), foreground_features.end());
    combined_features.insert(combined_features.end(), central_features.begin(), central_features.end());
    
    return combined_features;
}

std::vector<float> CombinedFeatures::computeResNetFeatures(const cv::Mat& img) {
    return resnet.getFeatures(img);
}

std::vector<float> CombinedFeatures::computeForegroundFeatures(const cv::Mat& img) {
    // Use color-texture histogram for foreground
    return computeColorTextureHist(img);
}

std::vector<float> CombinedFeatures::computeCentralRGBHist(const cv::Mat& img) {
    // Extract central region (50% of image)
    int centerX = img.cols / 4;
    int centerY = img.rows / 4;
    int width = img.cols / 2;
    int height = img.rows / 2;
    
    cv::Mat central = img(cv::Rect(centerX, centerY, width, height));
    return computeSplitRGBHist(central);
}

// Helper function to normalize a distance to [0,1] range
float CombinedFeatures::normalizeDistance(float distance, float max_dist) {
    return std::min(1.0f, std::max(0.0f, distance / max_dist));
}

// Helper function to compute L2 norm between feature vectors
float CombinedFeatures::computeL2Distance(const std::vector<float>& feat1, 
                                        const std::vector<float>& feat2) {
    float sum = 0.0f;
    for (size_t i = 0; i < feat1.size(); i++) {
        float diff = feat1[i] - feat2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float CombinedFeatures::computeDistance(const std::vector<float>& feat1, 
                                      const std::vector<float>& feat2) {
    // Split features into their components
    const int resnet_size = 512;
    const int foreground_size = 512;
    const int central_size = 768;
    
    // Extract individual feature vectors
    auto resnet_feat1 = std::vector<float>(feat1.begin(), 
                                         feat1.begin() + resnet_size);
    auto resnet_feat2 = std::vector<float>(feat2.begin(), 
                                         feat2.begin() + resnet_size);
    
    auto foreground_feat1 = std::vector<float>(feat1.begin() + resnet_size, 
                                             feat1.begin() + resnet_size + foreground_size);
    auto foreground_feat2 = std::vector<float>(feat2.begin() + resnet_size, 
                                             feat2.begin() + resnet_size + foreground_size);
    
    auto central_feat1 = std::vector<float>(feat1.begin() + resnet_size + foreground_size, 
                                          feat1.end());
    auto central_feat2 = std::vector<float>(feat2.begin() + resnet_size + foreground_size, 
                                          feat2.end());
    
    // Compute raw distances
    float resnet_dist = computeL2Distance(resnet_feat1, resnet_feat2);
    float foreground_dist = 1.0f - computeHistIntersection(foreground_feat1, foreground_feat2);
    float central_dist = 1.0f - computeHistIntersection(central_feat1, central_feat2);
    
    // Normalize distances
    // These max values should be tuned based on your dataset
    const float MAX_RESNET_DIST = 10.0f;      // Adjust based on typical ResNet distances
    const float MAX_FOREGROUND_DIST = 1.0f;   // Histogram intersection is already [0,1]
    const float MAX_CENTRAL_DIST = 1.0f;      // Histogram intersection is already [0,1]
    
    float norm_resnet_dist = normalizeDistance(resnet_dist, MAX_RESNET_DIST);
    float norm_foreground_dist = normalizeDistance(foreground_dist, MAX_FOREGROUND_DIST);
    float norm_central_dist = normalizeDistance(central_dist, MAX_CENTRAL_DIST);
    
    // Return weighted combination of normalized distances
    return RESNET_WEIGHT * norm_resnet_dist + 
           FOREGROUND_WEIGHT * norm_foreground_dist + 
           CENTER_WEIGHT * norm_central_dist;
}
