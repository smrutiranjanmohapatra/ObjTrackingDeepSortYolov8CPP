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
};

//// CLSCONF structure definition
//struct CLSCONF {
//    int cls;
//    float conf;
//};

//// DETECTION_ROW structure definition
//struct DETECTION_ROW1 {
//    VectorXf feature;  // Feature vector
//    Rect bounding_box; // Bounding box (x, y, width, height)
//
//    // Convert bounding box to tlwh format
//    VectorXf to_tlwh() const {
//        VectorXf tlwh(4);
//        tlwh << bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height;
//        return tlwh;
//    }
//};

// Type aliases
using DETECTIONS = vector<DETECTION_ROW>;
using DETECTIONSV2 = pair<vector<CLSCONF>, DETECTIONS>;

#endif // DETECTION_H
