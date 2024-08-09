#pragma once
#ifndef DETECTION_H
#define DETECTION_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "model.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

struct Detection {
    VectorXf feature;  // Feature vector (assumed size 256)
    Rect bounding_box; // Bounding box (x, y, width, height)
    float confidence;       // Confidence score
    int class_id;
};

struct TrackedObject {
    int track_id;
    Rect bounding_box;
    float confidence;
    int class_id;
   // VectorXf feature;  // This line to store feature vector
};

// Type aliases
using DETECTIONS = vector<DETECTION_ROW>;
using DETECTIONSV2 = pair<vector<CLSCONF>, DETECTIONS>;

#endif // DETECTION_H
