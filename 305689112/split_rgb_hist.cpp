/*
Yunyu Guo
CS5330 Project 2
Feb 2 2025
 
Split RGB Histogram Feature Extraction
 
Creates two 8x8x8 RGB histograms for top and bottom image halves
Each histogram uses 8 bins per color channel

Functions:
- computeSplitRGBHist: Creates normalized histograms from image halves
 */

#include <vector>
#include "opencv2/opencv.hpp"
#include "hist_util.h"

std::vector<float> computeSplitRGBHist(const cv::Mat& img) {
    const int bins = 8;  // bins per channel
    std::vector<float> features;
    features.reserve(2 * bins * bins * bins);  // Space for both histograms
    
    // Split image into top and bottom halves
    int mid_y = img.rows / 2;
    cv::Mat top_half = img(cv::Range(0, mid_y), cv::Range::all());//First Range: Specifies rows, Second Range: Specifies columns
    cv::Mat bottom_half = img(cv::Range(mid_y, img.rows), cv::Range::all());
    
    // Process each half
    for(const cv::Mat& half : {top_half, bottom_half}) {
        // Create 3D histogram for this half
        const int dims[] = {bins, bins, bins};  // Array specifying size in each dimension
        cv::Mat hist = cv::Mat::zeros(3, dims, CV_32F);  // 3 dimensions, size array, type
        
        // Process each pixel
        for(int y = 0; y < half.rows; y++) {
            for(int x = 0; x < half.cols; x++) {
                cv::Vec3b pixel = half.at<cv::Vec3b>(y, x);
                // Calculate bin indices
                int b_bin = pixel[0] * bins / 256;
                int g_bin = pixel[1] * bins / 256;
                int r_bin = pixel[2] * bins / 256;
                
                // Ensure indices are in range [0,7]
                b_bin = std::min(bins-1, b_bin);
                g_bin = std::min(bins-1, g_bin);
                r_bin = std::min(bins-1, r_bin);
                
                // Increment histogram bin
                hist.at<float>(b_bin, g_bin, r_bin) += 1.0f;
            }
        }
        
        // Normalize histogram
        hist /= (half.rows * half.cols);
        
        // Convert 3D histogram to 1D and add to features
        for(int r = 0; r < bins; r++) {
            for(int g = 0; g < bins; g++) {
                for(int b = 0; b < bins; b++) {
                    features.push_back(hist.at<float>(b, g, r));
                }
            }
        }
    }
    
    return features;
}
