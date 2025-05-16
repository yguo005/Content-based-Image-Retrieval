/*
Yunyu Guo
CS5330 Project 2
Jan 28 2025
Read the database of features from the CSV file
Compare the target image features with the database
Find the best matches
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include "opencv2/opencv.hpp"
#include "csv_util.h"
#include "feature_util.h"
#include "combined_features.h"
#include "chromaticity_hist.h"
#include "split_rgb_hist.h"
#include "color_texture_hist.h"
#include "resnet.h"


// Calculate Sum of Squared Differences between feature vectors
float calculateSSD(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    float ssd = 0.0f;
    for (size_t i = 0; i < feat1.size(); i++) {
        float diff = feat1[i] - feat2[i];
        ssd += diff * diff;
    }
    return ssd;
}

// Structure to hold image match results
struct ImageMatch {
    std::string filename;  
    float distance;
    
    // Define how < comparison works for ImageMatch objects
    bool operator< (const ImageMatch& other) const { //const ImageMatch& other: Reference to object being compared with; The const at the end: promises won't modify the current object on the left side of < 
        return distance < other.distance;
    }
};

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <target_image> <feature_database.csv> <num_matches> <feature_type>\n", argv[0]);
        printf("feature_type: 1 for center patch, 2 for chromaticity histogram, 3 for split RGB histogram, 4 for color-texture histogram, 5 for ResNet features, 6 for combined features\n");
        return -1;
    }

    // Read command line arguments
    std::string target_path = argv[1];
    std::string database_path = argv[2];
    int num_matches = atoi(argv[3]);
    int feature_type = atoi(argv[4]);

    // Read target image
    cv::Mat target = cv::imread(target_path);
    if (target.empty()) {
        printf("Cannot read target image %s\n", target_path.c_str());
        return -1;
    }

    // Initialize feature extractors
    CombinedFeatures combined;
    ResNetFeatures resnet("../ResNet18_olym.csv");

    // Extract target features
    std::vector<float> target_features;
    try {
        if (feature_type == 1) {
            target_features = extractCenterPatch(target);
        } else if (feature_type == 2) {
            target_features = computeChromaticityHist(target);
        } else if (feature_type == 3) {
            target_features = computeSplitRGBHist(target);
        } else if (feature_type == 4) {
            target_features = computeColorTextureHist(target);
        } else if (feature_type == 5) {
            target_features = resnet.getFeatures(target);
        } else if (feature_type == 6) {
            target_features = combined.computeFeatures(target);
        } else {
            printf("Invalid feature type\n");
            return -1;
        }
    } catch (const std::exception& e) {
        printf("Error extracting features: %s\n", e.what());
        return -1;
    }

    // Store matches
    std::vector<ImageMatch> matches;

    // Read database and compute distances
    std::vector<char*> filenames;
    std::vector<std::vector<float>> features;
    
    // Change from const string to char*
    char* db_path = strdup(database_path.c_str());  // Convert string to char*
    
    // Read the CSV file
    if (read_image_data_csv(db_path, filenames, features) != 0) {
        printf("Error reading database file %s\n", database_path.c_str());
        free(db_path);  // Free the allocated memory
        return -1;
    }
    
    free(db_path);  // Free the allocated memory

    // Compare with each database entry
    for (size_t i = 0; i < filenames.size(); i++) {
        float distance;
        if (feature_type == 6) {
            distance = combined.computeDistance(target_features, features[i]);
        } else {
            // Compute SSD for other feature types
            distance = 0.0f;
            for (size_t j = 0; j < target_features.size(); j++) {
                float diff = target_features[j] - features[i][j];
                distance += diff * diff;
            }
        }
        
        matches.push_back({std::string(filenames[i]), distance});
    }

    // Sort matches by distance
    std::sort(matches.begin(), matches.end());

    // Print top matches
    printf("Top %d matches:\n", num_matches);
    for (int i = 0; i < std::min(num_matches, (int)matches.size()); i++) {
        printf("%d. %s (distance: %.2f)\n", i + 1, 
               matches[i].filename.c_str(), matches[i].distance);
    }

    // Cleanup
    for (char* filename : filenames) {
        free(filename);
    }

    return 0;
}
