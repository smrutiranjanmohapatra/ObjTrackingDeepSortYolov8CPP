//#pragma once
//#ifndef IOU_H
//#define IOU_H
//
//#include <Eigen/Dense>
//#include <vector>
//
//// Function to compute the Intersection over Union (IoU) for a single bounding box
//// against a list of candidate bounding boxes
//Eigen::VectorXd computeIoU(const Eigen::VectorXd& bbox, const std::vector<Eigen::VectorXd>& candidates);
//
//// Function to compute the cost matrix based on IoU
//// The cost matrix is used to compute the matching cost between tracks and detections
//Eigen::MatrixXd iouCost(const std::vector<Eigen::VectorXd>& tracks,
//    const std::vector<Eigen::VectorXd>& detections);
//
//#endif // IOU_H
