
////This code used to work with random feature
//
//#include "deepsort.h"
//#include "tracker.h"
//#include "model.hpp"  // Assuming detection.cpp defines a Detection class
//#include "nn_matching.h"
//#include "linear_assignment.h"
//#include <vector>
//#include <iostream>
//
//using namespace std;
//
//// Constructor for DeepSORT
//DeepSORT::DeepSORT(float max_cosine_distance, int nn_budget,
//    float max_iou_distance, int max_age, int n_init)
//{
//    // Initialize the tracker with the given parameters
//    this->tracker_ = new tracker(max_cosine_distance, nn_budget,
//        max_iou_distance, max_age, n_init);
//}
//
//// Destructor for DeepSORT
//DeepSORT::~DeepSORT()
//{
//    delete this->tracker_;
//}
//
//// Method to update the tracker with new detections
//void DeepSORT::update(const vector<DetectBox>& detections)
//{
//    // Convert the detections to the format required by the tracker
//    DETECTIONS dets;
//    for (const auto& detection : detections) {
//        DETECTBOX box(detection.x1, detection.y1, detection.x2, detection.y2);
//        DETECTION_ROW row;
//        row.tlwh = box;
//        row.confidence = detection.confidence;
//        row.feature = VectorXf::Random(256);
//        dets.push_back(row);
//    }
//
//    this->tracker_->predict();
//    // Call the tracker update method
//    this->tracker_->update(dets);
//}
//
//// Method to update the tracker with new detections including class IDs and confidences
//void DeepSORT::update(const vector<DetectBox>& detections,  vector<CLSCONF>& cls_conf)
//{
//    // Convert the detections and class confidences to the format required by the tracker
//    /* vector<CLSCONF> clsConf;*/
//    DETECTIONS dets;
//
//    for (size_t i = 0; i < detections.size(); ++i) {
//        DETECTION_ROW row;
//        DETECTBOX bbox(detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2);
//        row.feature = VectorXf::Random(256);
//        // cout << "Feature for detection id :" << i << " -> " << row.feature.transpose() << endl;
//        row.tlwh = bbox;
//        row.confidence = detections[i].confidence;
//        dets.push_back(row);
//
//       CLSCONF conf;
//        conf.cls = cls_conf[i].cls;
//        conf.conf = cls_conf[i].conf;
//        cls_conf.push_back(conf);
//    }
//
//    DETECTIONSV2 dets_v2 = make_pair(cls_conf, dets);
//
//    this->tracker_->predict();
//    // Call the tracker update method
//    this->tracker_->update(dets_v2);
//}
//
//// Method to get the current list of tracked objects
//vector<TrackedObject> DeepSORT::getTrackedObjects()
//{
//    vector<TrackedObject> tracked_objects;
//
//    for (auto& track : this->tracker_->tracks) {
//        if (track.is_confirmed()) {
//            TrackedObject obj;
//            obj.track_id = track.track_id;
//            obj.bounding_box = Rect(track.to_tlwh()(0), track.to_tlwh()(1), track.to_tlwh()(3), track.to_tlwh()(2));
//            obj.confidence = track.conf;
//            obj.class_id = track.cls;
//           // obj.feature = track.features;
//            tracked_objects.push_back(obj);
//        }
//    }
//
//    return tracked_objects;
//}

#include "datatype.h"
#include "detection.h"
#include "deepsort.h"
#include "tracker.h"
#include "model.hpp"
#include "nn_matching.h"
#include "linear_assignment.h"
#include <vector>
#include <iostream>

using namespace std;
// Constructor for DeepSort
DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim,
    float max_cosine_distance, int nn_budget,
    float max_iou_distance, int max_age, int n_init)
    : enginePath(modelPath), batchSize(batchSize), featureDim(featureDim),
    maxCosineDist(max_cosine_distance), maxBudget(nn_budget),
    maxIouDist(max_iou_distance), maxAge(max_age), nInit(n_init),
    imgShape(cv::Size(64, 128))
{
    init();
}

// Destructor for DeepSort
DeepSort::~DeepSort()
{
    delete this->tracker_;
    delete featureExtractor;
}

// Initialization method
void DeepSort::init()
{
    // Initialize the tracker with the given parameters
    this->tracker_ = new tracker(maxCosineDist, maxBudget,maxIouDist, maxAge, nInit);
    //objTracker = new tracker(maxCosineDist, maxBudget, maxIouDist, maxAge, nInit);
    featureExtractor = new FeatureExtraction(batchSize, imgShape, featureDim);
    featureExtractor->loadOnnx(enginePath);
}

// Method to update the tracker with new detections
void DeepSort::update(cv::Mat& frame,const vector<DetectBox>& detections, cv::Point& crossLine)
{
    DETECTIONS dets = convertDetections(detections);
    if (!dets.empty()) {
        updateTracker(frame,dets,crossLine);
        processResults();
    }
}

// Method to update the tracker with new detections including class IDs and confidences
void DeepSort::update(cv::Mat& frame,const vector<DetectBox>& detections, vector<CLSCONF>& clsConf,cv::Point& crossLine)
{
    DETECTIONSV2 detectionsv2 = convertDetectionsWithClass(detections, clsConf);
    if (!detectionsv2.second.empty()) {
        updateTracker(frame,detectionsv2,crossLine);
        processResults();
    }
}

// Helper method to convert vector of DetectBox to DETECTIONS
DETECTIONS DeepSort::convertDetections(const vector<DetectBox>& detections)
{
    DETECTIONS dets;
    for (const auto& detection : detections) {
        DETECTBOX box(detection.x1, detection.y1, detection.x2, detection.y2);
        DETECTION_ROW row;
        row.tlwh = box;
        row.confidence = detection.confidence;
       
       // row.feature = VectorXf::Random(256);
        row.feature = VectorXf::Zero(2048);
        //row.feature = cv::Mat::zeros(featureDim, 1, CV_32F); // Placeholder for feature extraction
        dets.push_back(row);
    }
    return dets;
}

// Helper method to convert vector of DetectBox and class confidences to DETECTIONSV2
DETECTIONSV2 DeepSort::convertDetectionsWithClass(const vector<DetectBox>& detections, vector<CLSCONF>& clsConf)
{
    DETECTIONS dets;
    for (size_t i = 0; i < detections.size(); ++i) {
        DETECTBOX bbox(detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2);
        DETECTION_ROW row;
        row.tlwh = bbox;
        row.confidence = detections[i].confidence;
       
       // row.feature = VectorXf::Random(256);
        row.feature = VectorXf::Zero(2048);
        //row.feature = cv::Mat::zeros(featureDim, 1, CV_32F); // Placeholder for feature extraction
        dets.push_back(row);

        CLSCONF conf;
        conf.cls = clsConf[i].cls;
        conf.conf = clsConf[i].conf;
        clsConf.push_back(conf);
    }
    return make_pair(clsConf, dets);
}

// Helper method to update the tracker with the given detections
void DeepSort::updateTracker(cv::Mat& frame, DETECTIONS& detections, cv::Point& crossLine)
{

    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        this->tracker_->predict();
        this->tracker_->update(detections,crossLine);
    }
}

// Helper method to update the tracker with the given detections including class IDs
void DeepSort::updateTracker(cv::Mat& frame, DETECTIONSV2& detectionsv2,cv::Point& crossLine)
{
    DETECTIONS temp;
    temp = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, temp);
    if (flag) {
        detectionsv2.second = temp;
        this->tracker_->predict();
        this->tracker_->update(detectionsv2,crossLine);
    }
}

// Helper method to process the tracking results
void DeepSort::processResults()
{
    result.clear();
    results.clear();
    for (auto& track : this->tracker_->tracks) {
        if (track.is_confirmed() && track.time_since_update <= 1) {
            result.push_back(make_pair(track.track_id, cv::Rect(track.to_tlwh()(0), track.to_tlwh()(1), track.to_tlwh()(2), track.to_tlwh()(3))));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf), cv::Rect(track.to_tlwh()(0), track.to_tlwh()(1), track.to_tlwh()(2), track.to_tlwh()(3))));
            /*result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf), track.to_tlwh()));*/
        }
    }
}

// Method to get the current list of tracked objects
vector<TrackedObject> DeepSort::getTrackedObjects()
{
    vector<TrackedObject> tracked_objects;
    for (auto& track : this->tracker_->tracks) {
        if (track.is_confirmed()) {
            TrackedObject obj;
            obj.track_id = track.track_id;
            obj.bounding_box = Rect(track.to_tlwh()(0), track.to_tlwh()(1), track.to_tlwh()(3), track.to_tlwh()(2));
            obj.confidence = track.conf;
            obj.class_id = track.cls;
            tracked_objects.push_back(obj);
        }
    }
    return tracked_objects;
}
