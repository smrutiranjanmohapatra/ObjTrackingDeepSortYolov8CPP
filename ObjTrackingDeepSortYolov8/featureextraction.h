#pragma once
#ifndef FEATUREEXTRACTION_H
#define FEATUREEXTRACTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "datatype.h" 
#include "model.hpp"

using std::vector;

class FeatureExtraction {
public:
    FeatureExtraction(const int maxBatchSize, const cv::Size imgShape, const int featureDim);
    ~FeatureExtraction();

public:
    bool getRectsFeature(cv::Mat& img, DETECTIONS& det);
    bool getRectsFeature(DETECTIONS& det);
    void loadOnnx(std::string onnxPath);
    cv::Mat doInference(vector<cv::Mat>& imgMats);
    //cv::Mat doInference(cv::Mat& imgMat);

private:
   cv::Mat doInference_run(vector<cv::Mat> imgMats);
   //cv::Mat doInference_run(cv::Mat imgMat);
    void stream2det(cv::Mat stream, DETECTIONS& det);

private:
    cv::dnn::Net dnn_engine;
    const int maxBatchSize;
    const cv::Size imgShape;
    const int featureDim;
   // cv::Mat outputBuffer;


private:
    // int curBatchSize;
    //cv::Mat outputBuffer;
    //int inputIndex, outputIndex;
    //void* buffers[2];
    float means[3], std[3];
    const std::string inputName, outputName;
    
};

#endif // FEATUREEXTRACTION_H
