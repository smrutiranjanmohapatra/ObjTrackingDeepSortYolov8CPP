//
//
////Draw Two Lines
//#include "drawlines.h"
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//// Constructor
//DrawLines::DrawLines() {}
//
//// Mouse callback function
//void DrawLines::onMouse(int event, int x, int y, int, void* userdata) {
//    vector<Point>* pts = (vector<Point>*)userdata;
//    if (event == EVENT_LBUTTONDOWN) {
//        if (pts->size() == 0 || pts->size() == 2) {
//            // Store full (x, y) on the first and third clicks
//            pts->emplace_back(x, y);
//            cout << "Point selected: (" << x << ", " << y << ")" << endl;
//        }
//        else if (pts->size() == 1) {
//            // For the second click, store (x, y1)
//            pts->emplace_back(x, pts->at(0).y);
//            cout << "Point selected: (" << x << ", " << pts->at(0).y << ")" << endl;
//        }
//        else if (pts->size() == 3) {
//            // For the fourth click, store (x, y3)
//            pts->emplace_back(x, pts->at(2).y);
//            cout << "Point selected: (" << x << ", " << pts->at(2).y << ")" << endl;
//        }
//    }
//}
//
//// Function to draw two lines by capturing mouse clicks
//std::pair<std::pair<Point, Point>, std::pair<Point, Point>> DrawLines::drawTwoLines(Mat& img) {
//    cout << "Click to draw two lines on the image." << endl;
//    namedWindow("Draw Lines", WINDOW_AUTOSIZE);
//    points.clear();
//
//    setMouseCallback("Draw Lines", onMouse, &points);
//
//    while (points.size() < 4) {
//        imshow("Draw Lines", img);
//        waitKey(1);
//    }
//
//    // Draw the lines on the image
//    line(img, points[0], points[1], Scalar(0, 255, 0), 2); // Draw the first line (Green)
//    line(img, points[2], points[3], Scalar(255, 0, 0), 2); // Draw the second line (Blue)
//
//    // Show the final image with the lines drawn
//    imshow("Draw Lines", img);
//    waitKey(0);
//
//    destroyWindow("Draw Lines");
//
//    return { {points[0], points[1]}, {points[2], points[3]} };
//}
//


//Draw One Line 

#include "drawlines.h"
#include <iostream>

using namespace cv;
using namespace std;

// Constructor
DrawLines::DrawLines() {}

// Mouse callback function
void DrawLines::onMouse(int event, int x, int y, int, void* userdata) {
    vector<Point>* pts = (vector<Point>*)userdata;
    if (event == EVENT_LBUTTONDOWN) {
        if (pts->size() == 0) {
            // Store full (x, y) on the first click
            pts->emplace_back(x, y);
            cout << "Point selected: (" << x << ", " << y << ")" << endl;
        }
        else if (pts->size() == 1) {
            // Store (x, y1) on the second click to ensure same y-coordinate
            pts->emplace_back(x, pts->at(0).y);
            cout << "Point selected: (" << x << ", " << pts->at(0).y << ")" << endl;
        }
    }
}

// Function to draw one line with the same y-coordinate
std::pair<Point, Point> DrawLines::drawOneLine(Mat& img) {
    cout << "Click to draw one line on the image." << endl;
    namedWindow("Draw Line", WINDOW_AUTOSIZE);
    points.clear();

    setMouseCallback("Draw Line", onMouse, &points);

    while (points.size() < 2) {
        imshow("Draw Line", img);
        waitKey(1);
    }

    // Draw the line on the image
    line(img, points[0], points[1], Scalar(0, 255, 0), 2); // Draw the line (Green)

    // Show the final image with the line drawn
    imshow("Draw Line", img);
    waitKey(0);

    destroyWindow("Draw Line");

    return { points[0], points[1] };
}
