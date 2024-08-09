//
//#include "featureextraction.h"
//
//FeatureExtraction::FeatureExtraction(const int maxBatchSize, const cv::Size imgShape, const int featureDim)
//    : maxBatchSize(maxBatchSize), imgShape(imgShape), featureDim(featureDim) {}
//
//FeatureExtraction::~FeatureExtraction() {}
//
//void FeatureExtraction::loadOnnx(const std::string& onnxPath) {
//    dnn_engine = cv::dnn::readNetFromONNX(onnxPath);
//}
//
//bool FeatureExtraction::getRectsFeature(const cv::Mat& img, DETECTIONS& det) {
//    vector<cv::Mat> mats;
//    for (const auto& dbox : det) {
//        cv::Rect rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
//            int(dbox.tlwh(2)), int(dbox.tlwh(3)));
//        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
//        rect.width = rect.height * 0.5;
//        rect.x = std::max(rect.x, 0);
//        rect.y = std::max(rect.y, 0);
//        rect.width = std::min(rect.width, img.cols - rect.x);
//        rect.height = std::min(rect.height, img.rows - rect.y);
//
//        cv::Mat tempMat = img(rect).clone();
//        cv::resize(tempMat, tempMat, imgShape);
//        mats.push_back(tempMat);
//    }
//
//    outputBuffer = doInference(mats);
//    stream2det(outputBuffer, det);
//
//    return true;
//}
//
//cv::Mat FeatureExtraction::doInference(const vector<cv::Mat>& imgMats) {
//    cv::Mat blob = cv::dnn::blobFromImages(imgMats, 1.0, imgShape, cv::Scalar(0.485, 0.456, 0.406), true);
//    dnn_engine.setInput(blob);
//    return dnn_engine.forward();
//}
//
//void FeatureExtraction::stream2det(const cv::Mat& stream, DETECTIONS& det) {
//    int i = 0;
//    for (auto& dbox : det) {
//        for (int j = 0; j < featureDim; ++j) {
//            dbox.feature[j] = stream.at<float>(i, j);
//        }
//        ++i;
//    }
//}



#include "featureextraction.h"
#include <fstream>

using namespace std;

// Constructor
FeatureExtraction::FeatureExtraction(const int maxBatchSize, const cv::Size imgShape, const int featureDim)
    : maxBatchSize(maxBatchSize), imgShape(imgShape), featureDim(featureDim){
   // ,inputName("input"), outputName("output") {

    // Initialize normalization parameters
    means[0] = 0.485f; means[1] = 0.456f; means[2] = 0.406f;
    std[0] = 0.229f; std[1] = 0.224f; std[2] = 0.225f;
}

// Destructor
FeatureExtraction::~FeatureExtraction(){}

// Method to extract features from image patches based on detections
bool FeatureExtraction::getRectsFeature(cv::Mat& img, DETECTIONS& det) {
    std::vector<cv::Mat> mats;
    //cv::Mat mats;
    for (auto& dbox : det) {
        cv::Rect rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rect.x -= (rect.height * 0.5f - rect.width) * 0.5f;
        rect.width = rect.height * 0.5f;

        rect.x = std::max(rect.x, 0);
        if (rect.x > img.cols)
        {
            rect.x = img.cols-rect.width;
        }

        rect.y = std::max(rect.y, 0);
        if (rect.y > img.rows)
        {
            rect.y = img.rows-rect.height;
        }

        float tempwidth = rect.width;
        rect.width = std::min(rect.width, img.cols - rect.x);
        if (rect.width <= 0)
        {
            rect.width = tempwidth;
        }

        float tempheight = rect.height;
        rect.height = std::min(rect.height, img.rows - rect.y);
        if (rect.height <= 0)
        {
            rect.height = tempheight;
        }
        cv::Mat tempMat = img(rect).clone();
        cv::resize(tempMat, tempMat, imgShape);
        mats.push_back(tempMat);
    }
    if (mats.empty()) {
        return false;
    }
    cv::Mat out = doInference(mats);
    stream2det(out, det);  // Decode output to DETECTIONS
    return true;
}

// Dummy method to comply with previous interface (does nothing here)
bool FeatureExtraction::getRectsFeature(DETECTIONS& det) {
    return true;
}

// Method to load ONNX model for feature extraction
void FeatureExtraction::loadOnnx(std::string onnxPath) {
    dnn_engine = cv::dnn::readNetFromONNX(onnxPath);
    if (dnn_engine.empty()) {
        std::cerr << "Error: Could not load the ONNX model." <<std::endl;
        return;
    }

    //// Print layer names for debugging
    //vector<string> layerNames = dnn_engine.getLayerNames();
    //vector<int> outLayers = dnn_engine.getUnconnectedOutLayers();
    //std::cout << "Layer names:" << std::endl;
    //for (const auto& name : layerNames) {
    //    std::cout << name << std::endl;
    //}
   /* vector<string> outLayerNames;
    for (int i : outLayers)
        outLayerNames.push_back(layerNames[i - 1]);

    cout << "Input blobs: " << endl;
    for (const auto& layerName : layerNames)
        cout << layerName << endl;

    cout << "Output blobs: " << endl;
    for (const auto& layerName : outLayerNames)
        cout << layerName << endl;*/
}


// Method to perform inference on a batch of image patches
cv::Mat FeatureExtraction::doInference(std::vector<cv::Mat>& imgMats) {
    cv::Mat out;
    int mat_size = imgMats.size();
    if (mat_size > 0) {
        if (mat_size != maxBatchSize) {
            for (int i = 0; i < maxBatchSize - mat_size; ++i)
                imgMats.push_back(cv::Mat(imgShape, CV_8UC3, cv::Scalar(255, 255, 255)));  // Fill remaining batch with white images
        }

        out = doInference_run(imgMats);
    }

    return out;
}

//cv::Mat FeatureExtraction::doInference(cv::Mat& imgMat) {
//    cv::Mat out = doInference_run(imgMat);
//    return out;
//}


// Helper method to actually run the inference
cv::Mat FeatureExtraction::doInference_run(std::vector<cv::Mat> imgMats) {
    cv::Mat blob = cv::dnn::blobFromImages(imgMats, 1.0 / 255.0, imgShape, cv::Scalar(means[0], means[1], means[2]), true, false);
    dnn_engine.setInput(blob);
    //dnn_engine.setInput(blob, inputName);
    //cv::Mat out = dnn_engine.forward(outputName);
    cv::Mat out = dnn_engine.forward();
    return out;
}

//cv::Mat FeatureExtraction::doInference_run(cv::Mat imgMat) {
//    // Create a blob from the single image
//    cv::Mat blob = cv::dnn::blobFromImage(imgMat, 1.0 / 255.0, imgShape, cv::Scalar(means[0], means[1], means[2]), true, false);
//
//    // Set the blob as the input to the network
//    dnn_engine.setInput(blob);
//
//    // Run the forward pass to get the output
//    cv::Mat out = dnn_engine.forward();
//
//    return out;
//}


// Helper method to convert the output feature stream to DETECTIONS
void FeatureExtraction::stream2det(cv::Mat stream, DETECTIONS& det) {
    //cout <<"Matrix size : "<< stream.size() << endl;
    int i = 0;
    for (DETECTION_ROW& dbox : det) {
        for (int j = 0; j < featureDim; ++j) {
            float data = stream.at<float>(i, j);
            dbox.feature[j] = data;
        }
      //  cout << "Feature for detection id :" << i << " -> " << dbox.feature.transpose() << endl;
        i++;
    }
}
