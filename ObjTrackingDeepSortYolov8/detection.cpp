#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class ObjectDetection {

private:
    Net net;
    vector<string> classes;
    vector<Scalar> colors;
    float modelNMSThreshold;
    float modelConfidenceThreshold;
    float modelScoreThreshold;
    int image_size;
    Size2f modelShape{};
    set<string> targetClasses;
    bool letterBoxForSquare = true;
    int rows;
    int dimensions;

public:
    ObjectDetection(const string& onnx_path) {
        cout << "Loading Object Detection" << endl;
        cout << "Running opencv dnn with YOLOv8x ONNX" << endl;

        modelNMSThreshold = 0.5;
        modelConfidenceThreshold = 0.25;
        modelScoreThreshold = 0.45;
        //modelScoreThreshold = 0.8;
        image_size = 640;


        // Assuming Size2f is a typedef or class that expects float values
        modelShape = { static_cast<float>(image_size), static_cast<float>(image_size) };
        // modelShape = {image_size,image_size};

        // Load Network
        net = readNetFromONNX(onnx_path);
        if (net.empty()) {
            cerr << "Error: Could not load the neural network from the given path." << endl;
            return;
        }

        //Enable GPU CUDA if available
        if (cuda::getCudaEnabledDeviceCount() > 0)
        {
            cout << "\nRunning on CUDA" << endl;
            net.setPreferableBackend(DNN_BACKEND_CUDA);
            net.setPreferableTarget(DNN_TARGET_CUDA);
        }
        else
        {
            cout << "\nRunning on CPU" << endl;
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
        }

        //calling loadClassNames method for a pre defined classes file
        loadClassNames("D:\\OPENCV\\yolov8n\\classes.txt");

        //Set target classes as our requirements
       // targetClasses = { "car", "bus", "truck", "motorcycle","person","bicycle"};
        targetClasses = { "car", "bus", "truck", "motorcycle","bicycle" };
    }

    vector<string> loadClassNames(const string& classes_path) {
        ifstream ifs(classes_path.c_str());
        string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }

        // Generate random colors for each class
        RNG rng(0xFFFFFFFF);
        for (size_t i = 0; i < classes.size(); i++) {
            Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            colors.push_back(color);
        }

        return classes;
    }

    Mat formatToSquare(const Mat& source)
    {
        int col = source.cols;
        int row = source.rows;
        int _max = MAX(col, row);
        Mat result = Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(Rect(0, 0, col, row)));
        return result;
    }

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
        // Draw a bounding box.
        rectangle(frame, Point(left, top), Point(right, bottom), colors[classId], 3);

        // Get the label for the class name and its confidence
        string label = format("%.2f", conf);
        if (!classes.empty()) {
            CV_Assert(classId < (int)classes.size());
            label = classes[classId] + ":" + label;
        }

        // Display the label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar::all(255), FILLED);
        putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(), 1);
    }
    void postprocess(Mat& frame, vector<Mat>& outs, vector<int>& class_ids, vector<float>& confidences, vector<Rect>& boxes) {

        bool yolov8 = false;
        // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
        // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
        if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
        {
            yolov8 = true;
            rows = outs[0].size[2];
            dimensions = outs[0].size[1];

            outs[0] = outs[0].reshape(1, dimensions);
            transpose(outs[0], outs[0]);
        }
        float* data = (float*)outs[0].data;

        float x_factor = frame.cols / modelShape.width;
        float y_factor = frame.rows / modelShape.height;


        for (int i = 0; i < rows; ++i)
        {
            if (yolov8)
            {
                float* classes_scores = data + 4;

                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_idPoint;
                double maxClassScore;

                minMaxLoc(scores, 0, &maxClassScore, 0, &class_idPoint);

                if (maxClassScore > modelScoreThreshold)
                {

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    string className = classes[class_idPoint.x];
                    if (targetClasses.find(className) != targetClasses.end())
                    {
                        class_ids.push_back(class_idPoint.x);
                        confidences.push_back(maxClassScore);
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }
            else // yolov5
            {
                float confidence = data[4];
                cout << i << endl;
                if (confidence >= modelConfidenceThreshold)
                {
                    float* classes_scores = data + 5;

                    Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                    Point class_idPoint;
                    double max_class_score;

                    minMaxLoc(scores, 0, &max_class_score, 0, &class_idPoint);

                    if (max_class_score > modelScoreThreshold)
                    {

                        float x = data[0];
                        float y = data[1];
                        float w = data[2];
                        float h = data[3];

                        int left = int((x - 0.5 * w) * x_factor);
                        int top = int((y - 0.5 * h) * y_factor);

                        int width = int(w * x_factor);
                        int height = int(h * y_factor);

                        string className = classes[class_idPoint.x];
                        if (targetClasses.find(className) != targetClasses.end())
                        {
                            class_ids.push_back(class_idPoint.x);
                            confidences.push_back(max_class_score);
                            boxes.push_back(Rect(left, top, width, height));
                        }

                    }
                }
            }

            data += dimensions;
        }

        // Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences
        vector<int> indices;
        NMSBoxes(boxes, confidences, modelConfidenceThreshold, modelNMSThreshold, indices);
        vector<int> nmsClassIDs;
        vector<float> nmsConfidences;
        vector<Rect> nmsBoxes;
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            nmsClassIDs.push_back(class_ids[idx]);
            nmsConfidences.push_back(confidences[idx]);
            nmsBoxes.push_back(box);
            //drawPred(class_ids[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
        }

        class_ids.clear();
        class_ids = nmsClassIDs;
        confidences.clear();
        confidences = nmsConfidences;
        boxes.clear();
        boxes = nmsBoxes;
        // imshow("image", frame);
    }

    void detect(Mat& frame, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {

        Mat modelInput = frame;
        if (letterBoxForSquare && modelShape.width == modelShape.height)
            modelInput = formatToSquare(modelInput);

        Mat blob;
        blobFromImage(modelInput, blob, 1 / 255.0, Size(image_size, image_size), Scalar(), true, false);
        net.setInput(blob);

        // Run the forward pass to get output from the output layers
        vector<Mat> outs;
        //net.forward(outs, getOutputsNames(net));
        net.forward(outs, net.getUnconnectedOutLayersNames());
        if (outs.empty()) {
            cerr << "Error: No outputs obtained from the forward pass." << endl;
            return;
        }

        rows = outs[0].size[1];
        dimensions = outs[0].size[2];
        frame = modelInput;
        postprocess(frame, outs, classIds, confidences, boxes);
        //imshow("image2", frame);
    }

};

