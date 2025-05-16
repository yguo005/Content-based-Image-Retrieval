/*
Yunyu Guo
CS5330 Project 2
Jan 28 2025

This file implements the feature extraction function that:
1. Takes an input image
2. Extracts a 7x7 patch from its center
3. Returns the RGB values as a feature vector
 
The feature vector contains 147 values (7x7x3):
- For each pixel in the 7x7 patch
- Store Blue, Green, Red values
- Total size = 49 pixels * 3 channels = 147 values
*/

#include "feature_util.h"

std::vector<float> extractCenterPatch(const cv::Mat& img) {
    std::vector<float> features;
    
    // Get image dimensions
    int height = img.rows;
    int width = img.cols;
    
    // Calculate center position
    int startY = (height - 7) / 2;
    int startX = (width - 7) / 2;
    
    // Extract 7x7 patch with all color channels
    for (int y = 0; y < 7; y++) {
        for (int x = 0; x < 7; x++) {
            // Get BGR values for each pixel
            cv::Vec3b pixel = img.at<cv::Vec3b>(startY + y, startX + x);
            // Store all three color channels
            features.push_back(pixel[0]); // Blue
            features.push_back(pixel[1]); // Green
            features.push_back(pixel[2]); // Red
        }
    }
    
    return features;
}
