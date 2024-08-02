//#include "iou.h"
//#include <algorithm>
//#include <iostream>
//
//Eigen::VectorXd computeIoU(const Eigen::VectorXd& bbox, const std::vector<Eigen::VectorXd>& candidates) {
//    Eigen::VectorXd iou_scores(candidates.size());
//
//    Eigen::VectorXd bbox_tl = bbox.head(2);
//    Eigen::VectorXd bbox_br = bbox.head(2) + bbox.tail(2);
//
//    for (size_t i = 0; i < candidates.size(); ++i) {
//        Eigen::VectorXd candidate = candidates[i];
//        Eigen::VectorXd candidate_tl = candidate.head(2);
//        Eigen::VectorXd candidate_br = candidate.head(2) + candidate.tail(2);
//
//        Eigen::VectorXd tl = (bbox_tl.array().max(candidate_tl.array())).matrix();
//        Eigen::VectorXd br = (bbox_br.array().min(candidate_br.array())).matrix();
//        Eigen::VectorXd wh = (br.array() - tl.array()).matrix();
//        wh = wh.cwiseMax(0);
//
//        double area_intersection = wh.prod();
//        double area_bbox = bbox.tail(2).prod();
//        double area_candidate = candidate.tail(2).prod();
//
//        iou_scores[i] = area_intersection / (area_bbox + area_candidate - area_intersection);
//    }
//
//    return iou_scores;
//}
//
//Eigen::MatrixXd iouCost(const std::vector<Eigen::VectorXd>& tracks, const std::vector<Eigen::VectorXd>& detections) {
//    size_t num_tracks = tracks.size();
//    size_t num_detections = detections.size();
//    Eigen::MatrixXd cost_matrix(num_tracks, num_detections);
//
//    for (size_t i = 0; i < num_tracks; ++i) {
//        Eigen::VectorXd track = tracks[i];
//        std::vector<Eigen::VectorXd> detection_list(detections.begin(), detections.end());
//        Eigen::VectorXd iou_scores = computeIoU(track, detection_list);
//        cost_matrix.row(i) = 1.0 - iou_scores.transpose();
//    }
//
//    return cost_matrix;
//}
