/*
Yunyu Guo
CS5330 Project 2
Feb 2 2025


 Chromaticity Histogram Feature Extraction
 
 Creates a 16x16 histogram of r,g chromaticity values where: [0,1]
 r = R/(R+G+B), g = G/(R+G+B)
 
 Functions:
 - computeChromaticityHist: Creates normalized histogram from image
 
 Features are illumination-intensity invariant due to using color ratios
 */

#include <vector>
#include "opencv2/opencv.hpp"
#include "hist_util.h"

std::vector<float> computeChromaticityHist(const cv::Mat& img) {
    // Create 16x16 histogram 
    cv::Mat hist = cv::Mat::zeros(16, 16, CV_32F);
    
    // Process each pixel
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            float B = pixel[0];
            float G = pixel[1];
            float R = pixel[2];
            
            // Calculate r,g chromaticity, illumination-intensity invariant
            float sum = R + G + B;
            if(sum > 0) {  // Avoid division by zero
                float r = R/sum;  // r chromaticity
                float g = G/sum;  // g chromaticity
                
                // Calculate bin indices (r,g are in [0,1])
                const int bins = 16;
                int r_bin = std::min(bins-1, (int)(r * bins));
                int g_bin = std::min(bins-1, (int)(g * bins));
                
                // Increment histogram bin
                hist.at<float>(r_bin, g_bin) += 1.0f;
            }
        }
    }
    
    // Normalize histogram by total pixel count, makes histograms comparable regardless of image size
    hist /= (img.rows * img.cols);
    
    // Convert 2D histogram to 1D feature vector
    std::vector<float> features;
    features.reserve(256);  // Pre-allocate space
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            features.push_back(hist.at<float>(i, j));
        }
    }
    
    return features;
}