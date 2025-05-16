/*
Yunyu Guo
CS5330 Project 2
Feb 4 2025

Combined Color and Texture Histogram Feature Extraction
Creates two histograms:
1. RGB color histogram (8x8x8)
2. Gradient magnitude histogram (16 bins)
Features are weighted equally in distance computation
 */

#include <vector>
#include "opencv2/opencv.hpp"
#include "hist_util.h"

std::vector<float> computeColorTextureHist(const cv::Mat& img) {
    std::vector<float> features;
    const int color_bins = 8;   // bins per color channel
    const int grad_bins = 16;   // bins for gradient
    features.reserve(color_bins * color_bins * color_bins + grad_bins);
    
    // 1. Compute color histogram
    const int dims[] = {color_bins, color_bins, color_bins};
    cv::Mat color_hist = cv::Mat::zeros(3, dims, CV_32F);//Mat zeros(int ndims, const int* sizes, int type)
    
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            int b_bin = std::min(color_bins-1, pixel[0] * color_bins / 256);
            int g_bin = std::min(color_bins-1, pixel[1] * color_bins / 256);
            int r_bin = std::min(color_bins-1, pixel[2] * color_bins / 256);
            color_hist.at<float>(b_bin, g_bin, r_bin) += 1.0f;
        }
    }
    color_hist /= (img.rows * img.cols);  // Normalize
    
    // 2. Compute gradient magnitude histogram
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);// Convert BGR to grayscale
    
    // Compute Sobel gradients
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0);//1: first derivative in x direction measures how quickly pixel values are changing horizontally
    cv::Sobel(gray, grad_y, CV_32F, 0, 1);
    
    // Compute gradient magnitude
    //Magnitude: sqrt(grad_x^2 + grad_y^2), calculates the length of a 2D vector at each pixel 
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    // Create gradient histogram: 1D
    // 1. Create empty histogram with 16 bins
    cv::Mat grad_hist = cv::Mat::zeros(grad_bins, 1, CV_32F);//Mat zeros(int rows, int cols, int type);
    // 2. Find the maximum gradient magnitude
    double max_mag = 0;
    cv::minMaxLoc(magnitude, nullptr, &max_mag);//cv::minMaxLoc finds the minimum (don't need)and maximum values in a matrix
    
    //3. fill the histogram
    for(int y = 0; y < magnitude.rows; y++) {
        for(int x = 0; x < magnitude.cols; x++) {
            float mag = magnitude.at<float>(y, x);// accesses the gradient magnitude value at position (y,x) in the magnitude matrix
            int bin = std::min(grad_bins-1, (int)(mag * grad_bins / max_mag));
            grad_hist.at<float>(bin) += 1.0f;//accesses the count in a specific bin of gradient histogram; why use float: normalize the histogram later
        }
    }
    grad_hist /= (magnitude.rows * magnitude.cols);  // Normalize
    
    // Combine features into a single vector: first 512 values: Color histogram (8x8x8), last 16 values: Gradient histogram
    // 1. Add color histogram
    for(int r = 0; r < color_bins; r++) {
        for(int g = 0; g < color_bins; g++) {
            for(int b = 0; b < color_bins; b++) {
                features.push_back(color_hist.at<float>(b, g, r));
            }
        }
    }
    // 2. Add gradient histogram
    for(int i = 0; i < grad_bins; i++) {
        features.push_back(grad_hist.at<float>(i));
    }
    
    return features;
}

// Combined distance metric
float computeCombinedDistance(const std::vector<float>& hist1, 
                            const std::vector<float>& hist2) {
    const int color_bins = 8;
    const int grad_bins = 16;
    const int color_size = color_bins * color_bins * color_bins;
    
    // Split combined histograms into color and texture parts
    std::vector<float> color1(hist1.begin(), hist1.begin() + color_size);//+color_size: after 512 values
    std::vector<float> color2(hist2.begin(), hist2.begin() + color_size);
    std::vector<float> grad1(hist1.begin() + color_size, hist1.end());
    std::vector<float> grad2(hist2.begin() + color_size, hist2.end());
    
    // Compute distances separately
    float color_dist = 1.0f - computeHistIntersection(color1, color2);
    float grad_dist = 1.0f - computeHistIntersection(grad1, grad2);
    
    // Equal weighting
    return (color_dist + grad_dist) / 2.0f;
}
