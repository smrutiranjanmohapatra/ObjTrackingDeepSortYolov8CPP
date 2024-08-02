//#define _DLL_EXPORTS
//
//#include "deepsort.h"
//
////DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger) {
////    this->gpuID = gpuID;
////    this->enginePath = modelPath;
////    this->batchSize = batchSize;
////    this->featureDim = featureDim;
////    this->imgShape = cv::Size(64, 128);
////    this->maxBudget = 100;
////    this->maxCosineDist = 0.2;
////    this->gLogger = gLogger;
////    init();
////}
//
//
//DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID) {
//    this->gpuID = gpuID;
//    this->enginePath = modelPath;
//    this->batchSize = batchSize;
//    this->featureDim = featureDim;
//    this->imgShape = cv::Size(64, 128);
//    this->maxBudget = 100;
//    this->maxCosineDist = 0.2;
//    init();
//}
//void DeepSort::init() {
//    objTracker = new tracker(maxCosineDist, maxBudget);
//    //featureExtractor = new FeatureTensor(batchSize, imgShape, featureDim,gpuID,gLogger);
//    featureExtractor = new FeatureTensor(batchSize, imgShape, featureDim, gpuID);
//    /*int ret = enginePath.find(".onnx");
//    if (ret != -1)
//        featureExtractor->loadOnnx(enginePath);
//    else
//        featureExtractor->loadEngine(enginePath);*/
//
//    featureExtractor->loadOnnx(enginePath);
//}
//
//DeepSort::~DeepSort() {
//    delete objTracker;
//}
//
//void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
//    // preprocess Mat -> DETECTION
//    DETECTIONS detections;
//    vector<CLSCONF> clsConf;
//
//    for (DetectBox i : dets) {
//        DETECTBOX box(i.x1, i.y1, i.x2 - i.x1, i.y2 - i.y1);
//        DETECTION_ROW d;
//        d.tlwh = box;
//        d.confidence = i.confidence;
//        detections.push_back(d);
//        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
//    }
//    result.clear();
//    results.clear();
//    if (detections.size() > 0) {
//        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
//        sort(frame, detectionsv2);
//    }
//    // postprocess DETECTION -> Mat
//    dets.clear();
//    for (auto r : result) {
//        DETECTBOX i = r.second;
//        DetectBox b(i(0), i(1), i(2) + i(0), i(3) + i(1), 1.);
//        b.trackID = (float)r.first;
//        dets.push_back(b);
//    }
//    for (int i = 0; i < results.size(); ++i) {
//        CLSCONF c = results[i].first;
//        dets[i].classID = c.cls;
//        dets[i].confidence = c.conf;
//    }
//}
//
//
//void DeepSort::sort(cv::Mat& frame, DETECTIONS& detections) {
//    bool flag = featureExtractor->getRectsFeature(frame, detections);
//    if (flag) {
//        objTracker->predict();
//        objTracker->update(detections);
//        //result.clear();
//        for (Track& track : objTracker->tracks) {
//            if (!track.is_confirmed() || track.time_since_update > 1)
//                continue;
//            result.push_back(make_pair(track.track_id, track.to_tlwh()));
//        }
//    }
//}
//
//void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
//    std::vector<CLSCONF>& clsConf = detectionsv2.first;
//    DETECTIONS& detections = detectionsv2.second;
//    bool flag = featureExtractor->getRectsFeature(frame, detections);
//    if (flag) {
//        objTracker->predict();
//        objTracker->update(detectionsv2);
//        result.clear();
//        results.clear();
//        for (Track& track : objTracker->tracks) {
//            if (!track.is_confirmed() || track.time_since_update > 1)
//                continue;
//            result.push_back(make_pair(track.track_id, track.to_tlwh()));
//            results.push_back(make_pair(CLSCONF(track.cls, track.conf), track.to_tlwh()));
//        }
//    }
//}
//
//void DeepSort::sort(vector<DetectBox>& dets) {
//    DETECTIONS detections;
//    for (DetectBox i : dets) {
//        DETECTBOX box(i.x1, i.y1, i.x2 - i.x1, i.y2 - i.y1);
//        DETECTION_ROW d;
//        d.tlwh = box;
//        d.confidence = i.confidence;
//        detections.push_back(d);
//    }
//    if (detections.size() > 0)
//        sort(detections);
//    dets.clear();
//    for (auto r : result) {
//        DETECTBOX i = r.second;
//        DetectBox b(i(0), i(1), i(2), i(3), 1.);
//        b.trackID = r.first;
//        dets.push_back(b);
//    }
//}
//
//void DeepSort::sort(DETECTIONS& detections) {
//    bool flag = featureExtractor->getRectsFeature(detections);
//    if (flag) {
//        objTracker->predict();
//        objTracker->update(detections);
//        result.clear();
//        for (Track& track : objTracker->tracks) {
//            if (!track.is_confirmed() || track.time_since_update > 1)
//                continue;
//            result.push_back(make_pair(track.track_id, track.to_tlwh()));
//        }
//    }
//}
//
//#include "deepsort.h"
//#include "kalmanfilter.h"
//#include "iou.h"
//#include "linear_assignment.h"
//#include <algorithm>
//#include <numeric>
//
//DeepSORT::DeepSORT(int cascade_depth)
//    : cascade_depth_(cascade_depth), kf_(new KalmanFilter()) {}
//
//DeepSORT::~DeepSORT() {
//    delete kf_;
//}
//
//void DeepSORT::update(const std::vector<Bbox>& detections) {
//    std::vector<Bbox> tracks_to_update;
//    std::vector<int> track_indices;
//    std::vector<int> detection_indices(detections.size());
//    std::iota(detection_indices.begin(), detection_indices.end(), 0);
//
//    // Filter tracks that need to be updated
//    for (size_t i = 0; i < tracks_.size(); ++i) {
//        if (tracks_[i].time_since_update < 1) {
//            tracks_to_update.push_back(tracks_[i]);
//            track_indices.push_back(i);
//        }
//    }
//
//    // Compute cost matrix
//    auto [matches, unmatched_tracks, unmatched_detections] = matching_cascade(
//        iou_cost, 0.5, cascade_depth_, tracks_to_update, detections,
//        &track_indices, &detection_indices);
//
//    // Update matched tracks
//    for (const auto& match : matches) {
//        int track_idx = match.first;
//        int detection_idx = match.second;
//
//        Bbox& track = tracks_[track_idx];
//        const Bbox& detection = detections[detection_idx];
//
//        Eigen::VectorXd measurement(4);
//        measurement << detection.tlwh(0), detection.tlwh(1), detection.tlwh(2), detection.tlwh(3);
//        Eigen::VectorXd mean, covariance;
//        std::tie(mean, covariance) = kf_->predict(track.mean, track.covariance);
//        Eigen::VectorXd new_mean, new_covariance;
//        std::tie(new_mean, new_covariance) = kf_->update(mean, covariance, measurement);
//
//        track.mean = new_mean;
//        track.covariance = new_covariance;
//        track.time_since_update = 0;
//    }
//
//    // Create new tracks for unmatched detections
//    for (int idx : unmatched_detections) {
//        Bbox new_track;
//        Eigen::VectorXd measurement(4);
//        measurement << detections[idx].tlwh(0), detections[idx].tlwh(1), detections[idx].tlwh(2), detections[idx].tlwh(3);
//        Eigen::VectorXd mean;
//        Eigen::MatrixXd covariance;
//        std::tie(mean, covariance) = kf_->initiate(measurement);
//        new_track.mean = mean;
//        new_track.covariance = covariance;
//        new_track.time_since_update = 0;
//        tracks_.push_back(new_track);
//    }
//
//    // Increment time_since_update for tracks that were not matched
//    for (int idx : unmatched_tracks) {
//        tracks_[idx].time_since_update += 1;
//    }
//
//    // Remove old tracks
//    auto it = std::remove_if(tracks_.begin(), tracks_.end(),
//        [](const Bbox& track) { return track.time_since_update > 1; });
//    tracks_.erase(it, tracks_.end());
//}
//
//const std::vector<Bbox>& DeepSORT::get_tracks() const {
//    return tracks_;
//}

//
//#include "DeepSort.h"
//
//DeepSort::DeepSort(const string& feature_extractor_path) {
//    featureNet = dnn::readNetFromONNX(feature_extractor_path);
//    if (featureNet.empty()) {
//        cerr << "Error: Could not load the feature extractor network from the given path." << endl;
//    }
//}
//
//vector<float> DeepSort::extractFeatures(const Mat& objectImage) {
//    Mat blob;
//    dnn::blobFromImage(objectImage, blob, 1.0 / 255.0, Size(128, 128), Scalar(), true, false);
//    featureNet.setInput(blob);
//    Mat feature = featureNet.forward();
//    vector<float> features(feature.begin<float>(), feature.end<float>());
//    return features;
//}
//
//void DeepSort::update(const vector<Rect>& boxes, const vector<Mat>& features, Mat& frame) {
//    // Update tracking data structures with new detections and features
//    trackedBoxes = boxes;
//    trackFeatures.clear();
//    for (const auto& featureMat : features) {
//        trackFeatures.push_back(extractFeatures(featureMat));
//    }
//
//    // Simple association of tracks
//    associateTracks();
//}
//
//void DeepSort::associateTracks() {
//    // A simple method to associate tracks (this can be replaced with a more complex association algorithm)
//    // For example, matching based on feature similarity or using a distance metric
//    trackIDs.clear();
//    int trackID = 0;
//    for (const auto& box : trackedBoxes) {
//        trackIDs.push_back(trackID++);
//        tracks.push_back(box);
//        trackIDsList.push_back(trackID);
//    }
//}
//
//void DeepSort::drawTrackingResults(Mat& frame) {
//    for (size_t i = 0; i < trackedBoxes.size(); ++i) {
//        rectangle(frame, trackedBoxes[i], Scalar(0, 255, 0), 2);
//        putText(frame, to_string(trackIDs[i]), trackedBoxes[i].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
//    }
//}

#include "deepsort.h"
#include "tracker.h"
#include "model.hpp"  // Assuming detection.cpp defines a Detection class
#include "nn_matching.h"
#include "linear_assignment.h"
#include <vector>
#include <iostream>

using namespace std;

// Constructor for DeepSORT
DeepSORT::DeepSORT(float max_cosine_distance, int nn_budget,
    float max_iou_distance, int max_age, int n_init)
{
    // Initialize the tracker with the given parameters
    this->tracker_ = new tracker(max_cosine_distance, nn_budget,
        max_iou_distance, max_age, n_init);
}

// Destructor for DeepSORT
DeepSORT::~DeepSORT()
{
    delete this->tracker_;
}

// Method to update the tracker with new detections
void DeepSORT::update(const vector<DetectBox>& detections)
{
    // Convert the detections to the format required by the tracker
    DETECTIONS dets;
    for (const auto& detection : detections) {
        DETECTBOX box(detection.x1, detection.y1, detection.x2, detection.y2);
        DETECTION_ROW row;
        row.tlwh = box;
        row.confidence = detection.confidence;
        row.feature = VectorXf::Random(256);
        dets.push_back(row);
    }

    this->tracker_->predict();
    // Call the tracker update method
    this->tracker_->update(dets);
}

// Method to update the tracker with new detections including class IDs and confidences
void DeepSORT::update(const vector<DetectBox>& detections,  vector<CLSCONF>& cls_conf)
{
    // Convert the detections and class confidences to the format required by the tracker
    /* vector<CLSCONF> clsConf;*/
    DETECTIONS dets;

    for (size_t i = 0; i < detections.size(); ++i) {
        DETECTION_ROW row;
        DETECTBOX bbox(detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2);
        row.feature = VectorXf::Random(256);
        row.tlwh = bbox;
        row.confidence = detections[i].confidence;
        dets.push_back(row);

       CLSCONF conf;
        conf.cls = cls_conf[i].cls;
        conf.conf = cls_conf[i].conf;
        cls_conf.push_back(conf);
    }

    DETECTIONSV2 dets_v2 = make_pair(cls_conf, dets);

    this->tracker_->predict();
    // Call the tracker update method
    this->tracker_->update(dets_v2);
}

// Method to get the current list of tracked objects
vector<TrackedObject> DeepSORT::getTrackedObjects()
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

