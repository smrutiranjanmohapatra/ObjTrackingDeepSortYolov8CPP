//#pragma once
//#ifndef DEEPSORT_H
//#define DEEPSORT_H
//
//#ifdef _DLL_EXPORTS
//#define DLL_API _declspec(dllexport)
//#else
//#define DLL_API _declspec(dllimport)
//#endif
//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include "featuretensor.h"
//#include "tracker.h"
//#include "datatype.h"
//#include <vector>
//
//using std::vector;
////using nvinfer1::ILogger;
//
//class DeepSort {
//public:
//    //DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger);
//    DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID);
//    ~DeepSort();
//
//public:
//    void sort(cv::Mat& frame, vector<DetectBox>& dets);
//
//private:
//    void sort(cv::Mat& frame, DETECTIONS& detections);
//    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2);
//    void sort(vector<DetectBox>& dets);
//    void sort(DETECTIONS& detections);
//    void init();
//
//private:
//    std::string enginePath;
//    int batchSize;
//    int featureDim;
//    cv::Size imgShape;
//    float confThres;
//    float nmsThres;
//    int maxBudget;
//    float maxCosineDist;
//
//private:
//    vector<RESULT_DATA> result;
//    vector<std::pair<CLSCONF, DETECTBOX>> results;
//    tracker* objTracker;
//    FeatureTensor* featureExtractor;
//    int gpuID;
//};
//
//#endif  //deepsort.h



//
//#ifndef DEEPSORT_H
//#define DEEPSORT_H
//
//#include <vector>
//#include <Eigen/Dense>
//#include "kalmanfilter.h"
//#include "iou.h"
//#include "linear_assignment.h"
//
//// Define a struct or class for Bounding Box (Bbox)
//struct Bbox {
//    Eigen::VectorXd mean; // State vector (mean of the Kalman filter)
//    Eigen::MatrixXd covariance; // Covariance matrix of the Kalman filter
//    int time_since_update; // Time since the track was last updated
//    Eigen::VectorXd tlwh; // Bounding box in (top left x, top left y, width, height)
//};
//
//class DeepSORT {
//public:
//    // Constructor
//    DeepSORT(int cascade_depth = 3);
//
//    // Destructor
//    ~DeepSORT();
//
//    // Update tracks with new detections
//    void update(const std::vector<Bbox>& detections);
//
//    // Get current tracks
//    const std::vector<Bbox>& get_tracks() const;
//
//private:
//    int cascade_depth_; // The depth of the cascade for matching
//    KalmanFilter* kf_; // Pointer to the Kalman filter object
//
//    // List of tracks (predicted bounding boxes)
//    std::vector<Bbox> tracks_;
//};
//
//#endif // DEEPSORT_H


//
//#ifndef DEEPSORT_H
//#define DEEPSORT_H
//
//#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
//#include <vector>
//#include <list>
//#include <string>
//
//using namespace cv;
//using namespace std;
//
//class DeepSort {
//public:
//    DeepSort(const string& feature_extractor_path);
//    void update(const vector<Rect>& boxes, const vector<Mat>& features, Mat& frame);
//    void drawTrackingResults(Mat& frame);
//
//private:
//    // Feature extractor network
//    dnn::Net featureNet;
//
//    // Tracking data structures
//    vector<Rect> trackedBoxes;
//    vector<int> trackIDs;
//    vector<vector<float>> trackFeatures;
//    // A simple list to hold tracked objects
//    list<Rect> tracks;
//    list<int> trackIDsList;
//
//    // Helper functions
//    vector<float> extractFeatures(const Mat& objectImage);
//    void associateTracks();
//};
//
//#endif // DEEPSORT_H

//
//#ifndef DEEPSORT_H
//#define DEEPSORT_H
//
//#include <vector>
//#include "tracker.h"
//#include "detection.cpp"  // Assuming detection.cpp defines a Detection class
//
//// Forward declarations for types used
//class tracker;
//class Detection;
//class TrackedObject;
//
//using namespace std;
//
//// Structure to hold class ID and confidence
//struct CLSCONF {
//    int cls;  // Class ID
//    float conf;  // Confidence score
//};
//
//// Class to manage DeepSORT tracking
//class DeepSORT {
//public:
//    // Constructor
//    DeepSORT(float max_cosine_distance, int nn_budget,
//        float max_iou_distance, int max_age, int n_init);
//
//    // Destructor
//    ~DeepSORT();
//
//    // Method to update the tracker with new detections
//    void update(const vector<Detection>& detections);
//
//    // Method to update the tracker with new detections including class IDs and confidences
//    void update(const vector<Detection>& detections, const vector<CLSCONF>& cls_conf);
//
//    // Method to get the current list of tracked objects
//    vector<TrackedObject> getTrackedObjects();
//
//private:
//    // Pointer to the tracker instance
//    tracker* tracker_;
//
//    // Method to convert Detection to internal format
//    vector<DETECTION_ROW> convertDetections(const vector<Detection>& detections);
//
//    // Method to convert Detection and CLSCONF to internal format
//    pair<vector<CLSCONF>, vector<DETECTION_ROW>> convertDetectionsAndClsConf(
//        const vector<Detection>& detections, const vector<CLSCONF>& cls_conf);
//};
//
//#endif // DEEPSORT_H


#ifndef DEEPSORT_H
#define DEEPSORT_H

#include "tracker.h"
#include "detection.h"
#include <vector>

class DeepSORT {
public:
    DeepSORT(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init);
    ~DeepSORT();

    void update(const std::vector<DetectBox>& detections);
    void update(const std::vector<DetectBox>& detections,  std::vector<CLSCONF>& cls_conf);
    std::vector<TrackedObject> getTrackedObjects();

private:
    tracker* tracker_;
};

#endif // DEEPSORT_H

