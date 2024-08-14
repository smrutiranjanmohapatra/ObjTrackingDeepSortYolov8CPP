//#pragma once
//#ifndef DRAWLINES_H
//#define DRAWLINES_H
//
//#include <opencv2/opencv.hpp>
//#include <utility>
//
//class DrawLines {
//public:
//    // Constructor
//    DrawLines();
//
//    // Function to draw two lines by capturing mouse clicks
//    std::pair<std::pair<cv::Point, cv::Point>, std::pair<cv::Point, cv::Point>> drawTwoLines(cv::Mat& img);
//
//private:
//    std::vector<cv::Point> points;
//
//    // Mouse callback function for capturing clicks
//    static void onMouse(int event, int x, int y, int, void* userdata);
//};
//
//#endif // DRAWLINES_H

#ifndef DRAWLINES_H
#define DRAWLINES_H

#include <opencv2/opencv.hpp>
#include <vector>

class DrawLines {
public:
    DrawLines();  // Constructor

    // Function to draw one line with the same y-coordinate
    std::pair<cv::Point, cv::Point> drawOneLine(cv::Mat& img);

private:
    std::vector<cv::Point> points;

    // Mouse callback function for drawing lines
    static void onMouse(int event, int x, int y, int, void* userdata);
};

#endif // DRAWLINES_H
