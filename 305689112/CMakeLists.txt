cmake_minimum_required(VERSION 3.10)
project(project2)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set the path to ONNX Runtime include directory
include_directories(/opt/homebrew/Cellar/onnxruntime/1.20.1_2/include/onnxruntime)

# Set the path to ONNX Runtime library directory
link_directories(/opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib)

# Add source files
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Define common source files
set(COMMON_SOURCES
    ${SRC_DIR}/csv_util.cpp
    ${SRC_DIR}/feature_util.cpp
    ${SRC_DIR}/hist_util.cpp
    ${SRC_DIR}/histo.cpp
    ${SRC_DIR}/split_rgb_hist.cpp
    ${SRC_DIR}/color_texture_hist.cpp
    ${SRC_DIR}/resnet.cpp
    ${SRC_DIR}/combined_features.cpp
)

# Define source files for each executable
set(IMGMATCH_SOURCES
    ${SRC_DIR}/imgmatch.cpp
    ${COMMON_SOURCES}
)

set(FEATURE_SOURCES
    ${SRC_DIR}/feature.cpp
    ${COMMON_SOURCES}
)

# Add executables
add_executable(imgmatch ${IMGMATCH_SOURCES}) #Only includes the main() from imgmatch.cpp
add_executable(feature ${FEATURE_SOURCES})

# Include directories for header files
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${INCLUDE_DIR}
)

# Link OpenCV and ONNX Runtime libraries to both executables
target_link_libraries(imgmatch ${OpenCV_LIBS})
target_link_libraries(feature ${OpenCV_LIBS})

# Add combined features to both executables
target_sources(feature PRIVATE
    ${SRC_DIR}/combined_features.cpp
)

target_sources(imgmatch PRIVATE
    ${SRC_DIR}/combined_features.cpp
)

