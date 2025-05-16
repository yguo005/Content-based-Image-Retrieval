How to run:
1. cd to build Directory
2. run feature.cpp file: ./feature ../olympus/ features.csv 1 
(1 for center patch, 2 for chromaticity histogram, 3 for split RGB histogram, 4 for color texture histogram, 5 for ResNet features, 6 for combined features)
3. run imgmatch.cpp file: ./imgmatch ../olympus/pic.0164.jpg features.csv 5 1 
(5 for top 5 matches; 1 for center patch, 2 for chromaticity histogram, 3 for split RGB histogram, 4 for color texture histogram, 5 for ResNet features, 6 for combined features)

