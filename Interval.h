#ifndef TEST_OPENCV_INTERVAL_H
#define TEST_OPENCV_INTERVAL_H

#include "opencv2/core/types.hpp"

struct Interval
{
    int begin, end;
    int y_coordinate;
    int cluster_num;
    cv::Vec3b color;
    Interval *next;
    bool added;

    Interval();
    Interval(int begin, int end, int y_coordinate, int cluster_num, cv::Vec3b color);
};


#endif //TEST_OPENCV_INTERVAL_H
