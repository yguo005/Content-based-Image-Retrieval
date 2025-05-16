/*
 * Common histogram utilities
 */

#include "hist_util.h"

float computeHistIntersection(const std::vector<float>& hist1, const std::vector<float>& hist2) {
    float intersection = 0.0f;
    for(size_t i = 0; i < hist1.size(); i++) {
        intersection += std::min(hist1[i], hist2[i]);
    }
    return intersection;
}
