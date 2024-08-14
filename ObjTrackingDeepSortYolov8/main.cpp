//
//#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
//#include "detection.cpp" // Your ObjectDetection class
//#include "deepsort.h" // Assuming you have a Deep SORT class defined
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//// Function prototypes
//void drawTrackingResults(Mat& frame, const vector<int>& ids, const vector<Rect>& boxes, const vector<Scalar>& colors);
//
//int main() {
//    // Initialize the video capture
//    VideoCapture cap("path/to/your/video.mp4"); // Use 0 for webcam
//    if (!cap.isOpened()) {
//        cerr << "Error: Could not open video file or webcam." << endl;
//        return -1;
//    }
//
//    // Initialize object detection
//    ObjectDetection detector("path/to/yolov8.onnx");
//
//    // Initialize Deep SORT tracker
//    DeepSort deepSort; // You need to define this class
//
//    // Main loop
//    while (true) {
//        Mat frame;
//        cap >> frame;
//        if (frame.empty()) break; // Exit if the frame is empty
//
//        vector<int> classIds;
//        vector<float> confidences;
//        vector<Rect> boxes;
//
//        // Object detection
//        detector.detect(frame, classIds, confidences, boxes);
//
//        // Extract features from detected objects for Deep SORT
//        vector<Mat> features;
//        for (const auto& box : boxes) {
//            Mat cropped = frame(box);
//            Mat feature = deepSort.extractFeatures(cropped); // You need to define this method
//            features.push_back(feature);
//        }
//
//        // Update Deep SORT tracker
//        vector<int> ids = deepSort.update(features, boxes);
//
//        // Draw tracking results
//        drawTrackingResults(frame, ids, boxes, detector.colors);
//
//        // Display the frame
//        imshow("Object Tracking", frame);
//        if (waitKey(1) == 27) break; // Exit on 'Esc' key
//    }
//
//    cap.release();
//    destroyAllWindows();
//
//    return 0;
//}
//
//// Function to draw tracking results on the frame
//void drawTrackingResults(Mat& frame, const vector<int>& ids, const vector<Rect>& boxes, const vector<Scalar>& colors) {
//    for (size_t i = 0; i < ids.size(); ++i) {
//        rectangle(frame, boxes[i], colors[ids[i] % colors.size()], 2);
//        putText(frame, format("ID: %d", ids[i]), boxes[i].tl(), FONT_HERSHEY_SIMPLEX, 0.5, colors[ids[i] % colors.size()], 2);
//    }
//}


//
//#include <opencv2/opencv.hpp>
//#include <opencv2/dnn.hpp>
//#include <vector>
//#include "detection.cpp"
//#include "Deepsort.h"
//
//using namespace cv;
//using namespace cv::dnn;
//using namespace std;
//
//int main(int argc, char** argv) {
//    if (argc < 2) {
//        cerr << "Usage: " << argv[0] << "D:\\OPENCV\\yolov8n\\yolov8x.onnx" << endl;
//        return -1;
//    }
//
//    string onnxPath = argv[1];
//
//    // Initialize object detection
//    ObjectDetection objectDetection(onnxPath);
//
//    // Initialize DeepSORT
//    DeepSORT deepSort;
//
//    // Open video capture (change to a video file path if needed)
//    VideoCapture cap("D:\\OPENCV\\Images\\24.mp4");  // 0 for webcam or replace with video file path
//
//    if (!cap.isOpened()) {
//        cerr << "Error: Could not open video capture." << endl;
//        return -1;
//    }
//
//    Mat frame;
//    while (true) {
//        cap >> frame;
//        if (frame.empty()) {
//            break; // End of video stream
//        }
//
//        vector<int> classIds;
//        vector<float> confidences;
//        vector<Rect> boxes;
//
//        // Perform object detection
//        objectDetection.detect(frame, classIds, confidences, boxes);
//
//        // Prepare detections for DeepSORT
//        vector<Detection> detections;
//        for (size_t i = 0; i < classIds.size(); ++i) {
//            Rect bbox = boxes[i];
//            float confidence = confidences[i];
//            // Convert OpenCV Rect to DeepSORT BoundingBox format if needed
//            BoundingBox bboxDeepSort(bbox.x, bbox.y, bbox.width, bbox.height);
//            detections.push_back(Detection(bboxDeepSort, confidence));
//        }
//
//        // Update DeepSORT tracker
//        vector<Track> tracks = deepSort.update(detections);
//
//        // Draw the tracking results
//        for (const auto& track : tracks) {
//            Rect bbox(track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height);
//            Scalar color = Scalar(0, 255, 0); // Choose a color for the bounding box
//            rectangle(frame, bbox, color, 2);
//
//            // Draw the track ID
//            putText(frame, to_string(track.track_id), bbox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
//        }
//
//        // Show the frame with detections and tracks
//        imshow("Tracking", frame);
//
//        // Break the loop on 'q' key press
//        if (waitKey(1) == 'q') {
//            break;
//        }
//    }
//
//    // Release video capture
//    cap.release();
//    destroyAllWindows();
//
//    return 0;
//}


#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "deepsort.h"
#include "detection.h"
#include "detection.cpp"  // Assuming detection-related classes are defined here
#include "drawlines.h"

using namespace cv;
using namespace std;

// Function to calculate the center point of a bounding box
Point2f calculateCenterPoint(const Rect& bbox) {
    return Point2f(bbox.x + bbox.width * 0.5f, bbox.y + bbox.height * 0.5f);
}

// Function to check if a point has crossed a line
bool hasCrossedLine(const Point2f& point, const Point& lineStart,const Point& lineEnd) {
    // Check if the center y  point is on one the other side of the line
    return ((point.y < lineStart.y)&&(point.x>lineStart.x)&&(point.x<lineEnd.x));
}


// Main function
int main() {

    // Initialize ObjectDetection and DeepSORT
    ObjectDetection objectDetection("D:\\OPENCV\\yolov8n\\yolov8m.onnx");
    DeepSort deepSort("D:\\OPENCV\\fast-reid\\outputs\\onnx_model\\FastReIdModel.onnx", 1, 2048, 0.4, 100, 0.5, 70, 1);  // Example parameters
    

    // Load a video file
    VideoCapture sourceVideo("D:\\OPENCV\\Images\\23.mp4");
    // VideoCapture sourceVideo("C:\\Users\\ITLP 71\\Downloads\\Traffic IP Camera video.mp4");
    if (!sourceVideo.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }
   //cout << "Total frame : " << sourceVideo.get(CAP_PROP_FRAME_COUNT) << endl;

    Mat frame;
    sourceVideo.read(frame);
    resize(frame, frame, Size(1200, 800));

    // Draw two lines on the first frame
    DrawLines drawline;
    pair<Point, Point> singleLine = drawline.drawOneLine(frame);
    Point lineStart = singleLine.first;
    Point lineEnd = singleLine.second;

    //saving a video
    // Get video properties
    int frameWidth = static_cast<int>(frame.cols);
    int frameHeight = static_cast<int>(frame.rows);
    double framePerSec = sourceVideo.get(CAP_PROP_FPS);
    cout << "Frame width :"<<frameWidth << " height :" << frameHeight << "fbs : " << framePerSec << endl;
    // Initialize VideoWriter
    VideoWriter videoWriter;
    videoWriter.open("D:\\OPENCV\\ResultVideo\\output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), framePerSec, Size(frameWidth, frameHeight));
    //videoWriter.open("D:\\OPENCV\\ResultVideo\\output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), framePerSec, Size(1200, 800));
    //videoWriter.open("D:\\OPENCV\\ResultVideo\\output_video.mp4", VideoWriter::fourcc('X', '2', '6', '4'), framePerSec, Size(frameWidth, frameHeight), true);
    if (!videoWriter.isOpened()) {
        cerr << "Error opening video writer" << endl;
        return -1;
    }

    // Frame counter
    int frameCounter = 0;

    double fps = 0.0;
    std::chrono::time_point<std::chrono::steady_clock> startTime, endTime;
    //Start the timer
    startTime = std::chrono::steady_clock::now();
    while (sourceVideo.read(frame)) {
        if (frame.empty()) {
            break;
        }
       // frameCounter++;

        //// Skip every 5 frames, process only the 1st, 6th, 11th, etc.
        //if (frameCounter % 2 != 1) {
        //    continue;
        //}

        // Calculate FPS after each  frame
        endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        fps = 1.0 / elapsed.count();  // Calculate FPS based  on the time taken for the current frame
        // Reset the start time for the next frame
        startTime = endTime;

        resize(frame, frame, Size(1200, 800));

        // Display FPS on the frame
        // std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
        // Format FPS value with 2 decimal places
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << fps;
        std::string fpsText = "FPS: " + oss.str();
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);

        // Detect objects in the frame
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        objectDetection.detect(frame, classIds, confidences, boxes);
     
        // Prepare detections for DeepSORT
        vector<DetectBox> detections;
        vector<CLSCONF> cls_conf;
        //cout << "For Detection" << endl;
        //for (size_t i = 0; i < boxes.size(); ++i) {
        //    cout << "i : " << i << ", X : " << boxes[i].x << " Y: " << boxes[i].y << " H: " << boxes[i].height << " W: " << boxes[i].width << " ClassID: " << classIds[i] << " confidence: " << confidences[i] << endl;
        //    DetectBox detection(boxes[i].x,boxes[i].y,boxes[i].height,boxes[i].width);
        //    //// Assume the detection class has a method to extract features
        //    //detection.feature = objectDetection.extractFeature(frame, boxes[i]);

        //    CLSCONF conf;
        //    conf.cls = classIds[i];
        //    conf.conf = confidences[i];

        //    detections.push_back(detection);
        //    cls_conf.push_back(conf);
        //}

        cout << "For Detection" << endl;
        for (size_t i = 0; i < boxes.size(); ++i) {
            Point2f center = calculateCenterPoint(boxes[i]);
            // Draw the center point
            //circle(frame, center, 5, Scalar(0, 0, 255), -1);

            // Check if the center point has crossed either of the lines
            if (hasCrossedLine(center, lineStart,lineEnd)) {
                cout << "i : " << i << ", X : " << boxes[i].x << " Y: " << boxes[i].y << " H: " << boxes[i].height << " W: " << boxes[i].width << " ClassID: " << classIds[i] << " confidence: " << confidences[i] << endl;
                DetectBox detection(boxes[i].x, boxes[i].y, boxes[i].height, boxes[i].width);
                CLSCONF conf;
                conf.cls = classIds[i];
                conf.conf = confidences[i];

                detections.push_back(detection);
                cls_conf.push_back(conf);
            }
        }
        
        // Update the tracker with new detections
        deepSort.update(frame,detections,cls_conf,lineStart);

        // Get tracked objects
        vector<TrackedObject> trackedObjects = deepSort.getTrackedObjects();

        cout << "For Tracking " << endl;
        // Draw bounding boxes and tracking IDs
        for (const auto& obj : trackedObjects) {
            // Draw bounding box around the object
            rectangle(frame, obj.bounding_box, Scalar(0, 0, 0), 2);

            // Define text and calculate the size of the text box
            string label = "ID: " + to_string(obj.track_id);
            
            cout << "ID : " << obj.track_id << ", X : " << obj.bounding_box.x << " Y: " << obj.bounding_box.y << " H: " << obj.bounding_box.height << " W: " << obj.bounding_box.width << " ClassID: " << obj.class_id << " confidence: " << obj.confidence<<endl;
           // cout << "Feature for detection id :" << obj.track_id << " -> " << obj.feature << endl;
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseLine);

            // Calculate the top-left and bottom-right points for the rectangle
            Point topLeft(obj.bounding_box.x, obj.bounding_box.y - labelSize.height - baseLine);
            Point bottomRight(obj.bounding_box.x + labelSize.width, obj.bounding_box.y);

            // Draw filled rectangle to put the text inside
            rectangle(frame, topLeft, bottomRight, Scalar(255, 255, 255), FILLED);

            // Draw the text inside the rectangle
            putText(frame, label, Point(obj.bounding_box.x, obj.bounding_box.y - baseLine),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
           /* rectangle(frame, obj.bounding_box, Scalar(0, 0, 0), 1);
            putText(frame, "ID: " + to_string(obj.track_id), Point(obj.bounding_box.x, obj.bounding_box.y - 10),FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 1);*/
        }

        // Draw the line on the frame
        line(frame, lineStart, lineEnd, Scalar(0, 255, 0), 2);

        // Display the frame with detections and tracking results
        imshow("Detections and Tracking", frame);

        //resize(frame, frame, Size(1200, 800));
        // Write the frame to the output video
        videoWriter.write(frame);

        // Exit if ESC key is pressed
        int key = waitKey(1);
        if (key == 27) break;
    }

   
    sourceVideo.release();
    videoWriter.release();
    destroyAllWindows();
    return 0;
}
