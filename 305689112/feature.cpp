/*
Yunyu Guo
CS5330 Project 2
Jan 28 2025
This program processes a directory of images and extracts features from each image.
The features are then saved to a CSV file for later use in image matching.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "csv_util.h"
#include "feature_util.h"
#include "chromaticity_hist.h"
#include "split_rgb_hist.h"
#include "color_texture_hist.h"
#include "resnet.h"
#include "combined_features.h"


/*********************** Move to feature_util.cpp ********************************************/
/*// Extract 7x7 center patch features from an image
std::vector<float> extractCenterPatch(const cv::Mat& img) {//const: not to modify the input image, &: Reference to avoid copying the entire image,only passes the memory address of the original image
    std::vector<float> features; //float: calculate distances std results can exceed 255
    
    // Convert image to grayscale if it's not 
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {// Clone existing grayscale image
        gray = img.clone();
    }
    
    // Get image dimensions
    int height = gray.rows;
    int width = gray.cols;
    
    // Calculate center position
    int startY = (height - 7) / 2;
    int startX = (width - 7) / 2;
    
    // Extract 7x7 patch
    for (int y = 0; y < 7; y++) {
        for (int x = 0; x < 7; x++) {
            features.push_back(gray.at<uchar>(startY + y, startX + x));//startY + y: vertical position in full image; <uchar>: gets pixel value as unsigned char (0-255)
        }
    }
    
    return features;
} */



int main(int argc, char *argv[]) {
    bool first_file = true;// Used to determine if this is the first write to CSV

    if (argc != 4) {
        printf("Usage: %s <image_directory> <output_csv> <feature_type>\n", argv[0]);
        printf("feature_type: 1 for center patch, 2 for chromaticity histogram, 3 for split RGB histogram, 4 for color-texture histogram, 5 for ResNet features, 6 for combined features\n");
        return -1;
    }

    char *dirname = argv[1];
    char *output_csv = argv[2];
    int feature_type = atoi(argv[3]);
    
    DIR *dirp = opendir(dirname);// Returns directory stream pointer
    if (!dirp) {
        printf("Cannot open directory %s\n", dirname);
        return -1;
    }

    // Process each image in directory
    struct dirent *dp; //Declare directory entry pointer
    while ((dp = readdir(dirp)) != NULL) {
        if (strstr(dp->d_name, ".jpg") ||//returns pointer to ".jpg"
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {
            
            // Build full path
            char buffer[256];//Maximum path length of 255 characters
            strcpy(buffer, dirname);// Copy directory path to buffer
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);// Append filename
            
            // Read image and extract features
            cv::Mat img = cv::imread(buffer);
            if (img.empty()) continue;
            
            // Extract features based on chosen method
            std::vector<float> features;
            if (feature_type == 1) {
                features = extractCenterPatch(img);  // From feature_util.h
            } else if (feature_type == 2) {
                features = computeChromaticityHist(img);  // From chromaticity_hist.h
            } else if (feature_type == 3) {
                features = computeSplitRGBHist(img);  // From split_rgb_hist.h
            } else if (feature_type == 4) {
                features = computeColorTextureHist(img);  // From color_texture_hist.h
            } else if (feature_type == 5) {
                // For ResNet, just write empty features since they're pre-computed
                // The actual features will be read from ResNet18_olym.csv during matching
                features = std::vector<float>(512, 0.0f);  // 512 zeros
            } else if (feature_type == 6) {
                CombinedFeatures combined;  
                features = combined.computeFeatures(img);
            }
            
            // Append to CSV file, first call is true, will create file; sebsequent calls are false, will append to file
            append_image_data_csv(output_csv, buffer, features, first_file);
            first_file = false;
        }
    }
    
    closedir(dirp);
    return 0;
}
