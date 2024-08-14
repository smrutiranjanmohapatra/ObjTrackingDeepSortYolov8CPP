//#ifndef DEEPSORT_H
//#define DEEPSORT_H
//
//#include "tracker.h"
//#include "detection.h"
//#include <vector>
//
//class DeepSORT {
//public:
//    DeepSORT(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init);
//    ~DeepSORT();
//
//    // Method to update the tracker with new detections
//    void update(const std::vector<DetectBox>& detections);
//
//    // Method to update the tracker with new detections including class IDs and confidences
//    void update(const std::vector<DetectBox>& detections,  std::vector<CLSCONF>& cls_conf);
//
//    // Method to get the current list of tracked objects
//    std::vector<TrackedObject> getTrackedObjects();
//
//private:
//    tracker* tracker_;
//};
//
//#endif // DEEPSORT_H


#ifndef DEEPSORT_H
#define DEEPSORT_H

#include "detection.h"
#include "tracker.h"
#include "featureextraction.h"
#include "model.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// DeepSort class definition
class DeepSort {
public:
    // Constructor
    DeepSort(std::string modelPath, int batchSize, int featureDim,
        float max_cosine_distance, int nn_budget,
        float max_iou_distance, int max_age, int n_init);

    // Destructor
    ~DeepSort();

    // Method to update the tracker with new detections
    void update(cv::Mat& frame,const std::vector<DetectBox>& detections, cv::Point& crossLine);

    // Method to update the tracker with new detections including class IDs and confidences
    void update(cv::Mat& frame,const std::vector<DetectBox>& detections, std::vector<CLSCONF>& clsConf,cv::Point& crossLine);

    // Method to get the current list of tracked objects
    std::vector<TrackedObject> getTrackedObjects();

private:
    // Initialization method
    void init();

    // Helper method to convert vector of DetectBox to DETECTIONS
    DETECTIONS convertDetections(const std::vector<DetectBox>& detections);

    // Helper method to convert vector of DetectBox and class confidences to DETECTIONSV2
    DETECTIONSV2 convertDetectionsWithClass(const std::vector<DetectBox>& detections, std::vector<CLSCONF>& clsConf);

    // Helper method to update the tracker with the given detections
    void updateTracker(cv::Mat& frame, DETECTIONS& detections, cv::Point& crossLine);

    // Helper method to update the tracker with the given detections including class IDs
    void updateTracker(cv::Mat& frame, DETECTIONSV2& detectionsv2,cv::Point& crossLine);

    // Helper method to process the tracking results
    void processResults();

    // Private members
    std::string enginePath;  // Path to the ONNX model
    int batchSize;           // Batch size for feature extraction
    int featureDim;          // Feature dimension
    float maxCosineDist;     // Maximum cosine distance for tracker
    int maxBudget;           // Maximum budget for the tracker
    float maxIouDist;        // Maximum IoU distance for tracker
    int maxAge;              // Maximum age for track
    int nInit;               // Number of frames to wait before confirming a track
    cv::Size imgShape;       // Shape of the input image for feature extraction

    tracker* tracker_;         // Pointer to the tracker object
    FeatureExtraction* featureExtractor; // Pointer to the feature extraction object

    // Tracking results
    std::vector<std::pair<int, cv::Rect>> result;
    std::vector<std::pair<CLSCONF, cv::Rect>> results;
};

#endif // DEEPSORT_H
