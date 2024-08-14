#include "tracker.h"
#include "nn_matching.h"
#include "linear_assignment.h"
using namespace std;

#define MY_inner_DEBUG
#ifdef MY_inner_DEBUG
#include <string>
#include <iostream>
#endif
using namespace std;
tracker::tracker(               /*NearNeighborDisMetric *metric, */
    float max_cosine_distance, int nn_budget,
    float max_iou_distance, int max_age, int n_init)
{
    //max_cosine_distance: The maximum allowable distance for associating detections with tracks.
    //nn_budget : The budget for storing features in the metric.
    this->metric = new NearNeighborDisMetric(
        NearNeighborDisMetric::METRIC_TYPE::cosine,
        max_cosine_distance, nn_budget);
    this->max_iou_distance = max_iou_distance;     //Maximum distance (in IOU) for associating detections with tracks
    this->max_age = max_age;                       //The maximum number of frames a track can exist without being updated before it's deleted.
    this->n_init = n_init;                         // The number of consecutive frames in which a detection must appear before it's confirmed as a track.

    this->kf = new abc::KalmanFilter();
    this->tracks.clear();
  
    this->_next_idx = 1;                         // This index is used to assign unique IDs to new tracks.
}

void tracker::predict()
{
    for (Track& track : tracks) {
        track.predit(kf);
    }
}

void tracker::update(const DETECTIONS& detections,cv::Point& crossLine)
{
    TRACKER_MATCHD res;
    _match(detections, res,crossLine);

    vector < MATCH_DATA >& matches = res.matches;
    for (MATCH_DATA& data : matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx]);
    }

    vector < int >& unmatched_tracks = res.unmatched_tracks;
    for (int& track_idx : unmatched_tracks) {
        this->tracks[track_idx].mark_missed();
    }

    vector < int >& unmatched_detections = res.unmatched_detections;
    for (int& detection_idx : unmatched_detections) {
        this->_initiate_track(detections[detection_idx]);
    }

    vector<int>& scope_out_tracks = res.scope_out_tracks;
    for (int& track_idx : scope_out_tracks) {
        this->tracks[track_idx].scope_missed();
    }

    vector < Track >::iterator it;
    for (it = tracks.begin(); it != tracks.end();) {
        if ((*it).is_deleted()) it = tracks.erase(it);
        else ++it;
    }
    vector < int >active_targets;
    vector < TRACKER_DATA > tid_features;
    for (Track& track : tracks) {
        if (track.is_confirmed() == false) continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        //FEATURESS t = FEATURESS(0, 256);
        FEATURESS t = FEATURESS(0, 2048);
        track.features = t;
    }
    this->metric->partial_fit(tid_features, active_targets);
}

void tracker::update(const DETECTIONSV2& detectionsv2,cv::Point& crossLine)
{
    const vector<CLSCONF>& clsConf = detectionsv2.first;
    const DETECTIONS& detections = detectionsv2.second;
    TRACKER_MATCHD res;
    _match(detections, res,crossLine);

    vector < MATCH_DATA >& matches = res.matches;
    for (MATCH_DATA& data : matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx], clsConf[detection_idx]);
    }

    vector < int >& unmatched_tracks = res.unmatched_tracks;
    for (int& track_idx : unmatched_tracks) {
        this->tracks[track_idx].mark_missed();
        cout << "Unmatched_Tracks  : ID : " << this->tracks[track_idx].track_id <<
            " ,x : " << this->tracks[track_idx].to_tlwh()(0) << " ,y : " << this->tracks[track_idx].to_tlwh()(1) <<
            " ,w : " << this->tracks[track_idx].to_tlwh()(2) << " ,h : " << this->tracks[track_idx].to_tlwh()(3) << endl;
    }

    vector < int >& unmatched_detections = res.unmatched_detections;
    for (int& detection_idx : unmatched_detections) {
        this->_initiate_track(detections[detection_idx], clsConf[detection_idx]);
    }
    
    vector<int>& scope_out_tracks = res.scope_out_tracks;
    for (auto track_idx : scope_out_tracks) {

        for (auto& it:tracks) {
            if (it.track_id == track_idx)
            {
                cout << "Delating ID OutOf_Scope : ID : " << it.track_id <<
                    " ,x : " << it.to_tlwh()(0) << " ,y : " << it.to_tlwh()(1) <<
                    " ,w : " << it.to_tlwh()(2) << " ,h : " << it.to_tlwh()(3) << endl;
                it.scope_missed();
            }
        }
       // this->tracks[track_idx].scope_missed();
    }
    vector < Track >::iterator it;
    for (it = tracks.begin(); it != tracks.end();) {
        if ((*it).is_deleted())
        {
            cout << "ID : " << it->track_id << " deleted Successfully" << endl;
            it = tracks.erase(it);
        }

        else ++it;
    }
    vector < int >active_targets;
    vector < TRACKER_DATA > tid_features;
    for (Track& track : tracks) {
        if (track.is_confirmed() == false) continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        FEATURESS t = FEATURESS(0, Feature_Vector_Dim);     //creates an empty feature matrix with no rows and 256 columns.
        track.features = t;                  //clears the feature data for the track after the current features have been processed and stored.
    }
    this->metric->partial_fit(tid_features, active_targets);
}

void tracker::_match(const DETECTIONS& detections, TRACKER_MATCHD& res,cv::Point& crossLine)
{
    vector < int >confirmed_tracks;
    vector < int >unconfirmed_tracks;
    vector < int >out_of_Scope_tracks;

    int idx = 0;
    for (Track& t : tracks) {
        if (t.is_confirmed()) 
            confirmed_tracks.push_back(idx);
        else 
            unconfirmed_tracks.push_back(idx);
        idx++;
    }
    // Perform cascade matching for confirmed tracks
    TRACKER_MATCHD matcha = linear_assignment::getInstance()->matching_cascade(
        this, &tracker::gated_matric,
        this->metric->mating_threshold,
        this->max_age,
        this->tracks,
        detections,
        confirmed_tracks);
    // Prepare IOU track candidates from unconfirmed tracks
    vector < int >iou_track_candidates;
    iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
    vector < int >::iterator it;
    for (it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();) {
        int idx = *it;
        if (tracks[idx].time_since_update == 1) {       //push into unconfirmed
            iou_track_candidates.push_back(idx);
            it = matcha.unmatched_tracks.erase(it);
            continue;
        }
        ++it;
    }
    // Perform IOU matching for remaining tracks
    TRACKER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
        this, &tracker::iou_cost,
        this->max_iou_distance,
        this->tracks,
        detections,
        iou_track_candidates,
        matcha.unmatched_detections);

    //get result:
    // Combine matches from both cascade and IOU matching
    res.matches.assign(matcha.matches.begin(), matcha.matches.end());
    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
    //unmatched_tracks;
    res.unmatched_tracks.assign(matcha.unmatched_tracks.begin(),matcha.unmatched_tracks.end());
    res.unmatched_tracks.insert(res.unmatched_tracks.end(),matchb.unmatched_tracks.begin(), matchb.unmatched_tracks.end());
    res.unmatched_detections.assign(matchb.unmatched_detections.begin(),matchb.unmatched_detections.end());

    //Store Out of scope trackers
    _outOfScope(out_of_Scope_tracks,crossLine);
    res.scope_out_tracks.assign(out_of_Scope_tracks.begin(), out_of_Scope_tracks.end());
    cout << "Out of scope size :" << out_of_Scope_tracks.size() << endl;
}

void tracker::_initiate_track(const DETECTION_ROW& detection)
{
    KAL_DATA data = kf->initiate(detection.to_xyah());
    KAL_MEAN mean = data.first;
    KAL_COVA covariance = data.second;

    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->n_init,
        this->max_age, detection.feature));
    _next_idx += 1;
}
void tracker::_initiate_track(const DETECTION_ROW& detection, CLSCONF clsConf)
{
    KAL_DATA data = kf->initiate(detection.to_xyah());
    KAL_MEAN mean = data.first;
    KAL_COVA covariance = data.second;

    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->n_init,
        this->max_age, detection.feature, clsConf.cls, clsConf.conf));
    _next_idx += 1;
}


void tracker::_outOfScope(vector <int>& out_of_Scope_tracks)
{
    //finding out of scope ids with frame rows cols weight height 
    for (auto it : tracks)
    {
        DetectBox temp;
        temp.x1 = it.to_tlwh()(0);
        temp.y1 = it.to_tlwh()(1);
        temp.x2 = it.to_tlwh()(2); 
        temp.y2 = it.to_tlwh()(3);
        temp.trackID = it.track_id;
        if (temp.x1 <= 0 ||temp.x1 >= 1200 ||
            temp.y1 <= 0 || temp.y2 >= 800 ||
            temp.x2 <= 15|| temp.y2 <=15)
        {
            out_of_Scope_tracks.push_back(temp.trackID);
            //cout << "Track ID Out of Scope  :" << temp.trackID << endl;
        }
    }
}

//OutofScope ID after crossing the line or height width srink 
void tracker::_outOfScope(vector <int>& out_of_Scope_tracks,cv::Point& crossLine)
{
    for (auto it : tracks)
    {
        DetectBox temp;
        temp.y1 = it.to_tlwh()(1);
        temp.x2 = it.to_tlwh()(2);
        temp.y2 = it.to_tlwh()(3);
        temp.trackID = it.track_id;
        if (temp.y1 > crossLine.y|| temp.x2 <= 10 || temp.y2 <= 10)
        {
            out_of_Scope_tracks.push_back(temp.trackID);
            //cout << "Track ID Out of Scope  :" << temp.trackID << endl;
        }
    }
}

DYNAMICM tracker::gated_matric(
    std::vector < Track >& tracks,
    const DETECTIONS& dets,
    const std::vector < int >& track_indices,
    const std::vector < int >& detection_indices)
{
   // FEATURESS features(detection_indices.size(), 256);
    FEATURESS features(detection_indices.size(), 2048);
    int pos = 0;
    for (int i : detection_indices) {
        features.row(pos++) = dets[i].feature;
    }
    vector < int >targets;
    for (int i : track_indices) {
        targets.push_back(tracks[i].track_id);
    }
    DYNAMICM cost_matrix = this->metric->distance(features, targets);
    DYNAMICM res = linear_assignment::getInstance()->gate_cost_matrix(
        this->kf, cost_matrix, tracks, dets, track_indices,
        detection_indices);
    return res;
}

DYNAMICM
tracker::iou_cost(
    std::vector < Track >& tracks,
    const DETECTIONS& dets,
    const std::vector < int >& track_indices,
    const std::vector < int >& detection_indices)
{
    //!!!python diff: track_indices && detection_indices will never be None.
    //    if(track_indices.empty() == true) {
    //        for(size_t i = 0; i < tracks.size(); i++) {
    //            track_indices.push_back(i);
    //        }
    //    }
    //    if(detection_indices.empty() == true) {
    //        for(size_t i = 0; i < dets.size(); i++) {
    //            detection_indices.push_back(i);
    //        }
    //    }
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for (int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        if (tracks[track_idx].time_since_update > 1) {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
            continue;
        }
        DETECTBOX bbox = tracks[track_idx].to_tlwh();
        int csize = detection_indices.size();
        DETECTBOXSS candidates(csize, 4);
        for (int k = 0; k < csize; k++) candidates.row(k) = dets[detection_indices[k]].tlwh;
        Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
        cost_matrix.row(i) = rowV;
    }
    return cost_matrix;
}

Eigen::VectorXf
tracker::iou(DETECTBOX& bbox, DETECTBOXSS& candidates)
{
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];

    Eigen::Matrix < float, -1, 2 > candidates_tl;
    Eigen::Matrix < float, -1, 2 > candidates_br;
    candidates_tl = candidates.leftCols(2);
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    //    Eigen::VectorXf area_intersection(size);
    //    Eigen::VectorXf area_candidates(size);
    Eigen::VectorXf res(size);
    for (int i = 0; i < size; i++) {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1; w = (w < 0 ? 0 : w);
        float h = br_2 - tl_2; h = (h < 0 ? 0 : h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection / (area_bbox + area_candidates - area_intersection);
    }
    return res;
}