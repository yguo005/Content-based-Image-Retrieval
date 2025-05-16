# Project 2: Content-based Image Retrieval

## Development Environment
    - OpenCV Version: [e.g., OpenCV 4.x (as used in Project 1)]


## Project Overview
This project implements a content-based image retrieval (CBIR) system. Given a target image and a database of images, the system finds and ranks images from the database that are most similar to the target image based on various visual features. Implemented features include baseline 7x7 pixel matching, color histograms, multi-histograms, combined texture and color features, and deep network embeddings (ResNet18). The project explores different feature extraction methods and distance metrics.

## Files Submitted
- `main.cpp` (or your primary executable's source file, e.g., `cbir.cpp`)
- `features.cpp`
- `features.h`
- `csv_util.cpp` (if you used/modified the provided CSV utility)
- `csv_util.h` (if you used/modified the provided CSV utility)
- `Makefile` (if used)
- `readme.md`


## Compilation and Execution

### Compilation
- If using a Makefile: `make`
- Example command for manual compilation (adjust as needed):
  `g++ -o cbir main.cpp features.cpp csv_util.cpp \`pkg-config --cflags --libs opencv4\``

### Running the Executable(s)

The system can be run as a command-line program.
**General Command Structure:**
`./cbir <target_image_path> <image_database_directory> <feature_type> <distance_metric_or_matching_method> <N_results> [optional: path_to_precomputed_features.csv]`

**Example Usage:**

1.  **Baseline Matching:**
    `./cbir ../olympus/pic.1016.jpg ../olympus/ baseline ssd 5`
    *(Feature Type: `baseline`, Distance Metric: `ssd`)*

2.  **Histogram Matching (e.g., 2D RG histogram with 16 bins per channel):**
    `./cbir ../olympus/pic.0164.jpg ../olympus/ histogram_rg16 intersection 5`
    *(Feature Type: `histogram_rg16` (or your chosen name, e.g., `histogram_rgb8`), Distance Metric: `intersection`)*

3.  **Multi-histogram Matching (e.g., top/bottom halves, RGB 8 bins):**
    `./cbir ../olympus/pic.0274.jpg ../olympus/ multihist_tb_rgb8 weighted_intersection 5`
    *(Feature Type: `multihist_tb_rgb8`, Distance Metric: `weighted_intersection` (or your design))*

4.  **Texture and Color Matching (e.g., Sobel magnitude histogram + RGB histogram):**
    `./cbir ../olympus/pic.0535.jpg ../olympus/ texture_color_sobel_rgb8 combined_metric 5`
    *(Feature Type: `texture_color_sobel_rgb8`, Distance Metric: `combined_metric` (your design))*

5.  **Deep Network Embeddings (using precomputed features):**
    `./cbir ../olympus/pic.0893.jpg ../olympus/ dnn cosine 5 resnet_features.csv`
    *(Feature Type: `dnn`, Distance Metric: `cosine` or `ssd`. The last argument is the path to your CSV file containing pre-extracted ResNet18 features as per Task 5. For target images like pic.0893.jpg, its features must be present in this CSV.)*

**Note on Two-Program Approach (if implemented for Task 1, option 2):**
If split feature extraction and matching:
*   **Feature Extraction Program:**
    `./extract_features <image_database_directory> <feature_type> <output_feature_file.csv>`
    Example: `./extract_features ../olympus/ histogram_rgb8 features_rgb8.csv`
*   **Matching Program:**
    `./match_image <target_image_path> <feature_type> <feature_file.csv> <distance_metric> <N_results>`
    Example: `./match_image ../olympus/pic.0164.jpg histogram_rgb8 features_rgb8.csv intersection 5`



**Implemented Feature Types:**

- `baseline`: 7x7 center square
- `histogram_[your_spec]`: e.g., `histogram_rg16`, `histogram_rgb8`
- `multihist_[your_spec]`: e.g., `multihist_topbottom_rgb8`
- `texture_color_[your_spec]`: e.g., `texture_color_sobelmag_rgb8`
- `dnn`: ResNet18 embeddings

**Implemented Distance Metrics / Matching Methods:**

- `ssd`: Sum of Squared Differences
- `intersection`: Histogram Intersection
- `cosine`: Cosine Distance (1 - cosine similarity)





