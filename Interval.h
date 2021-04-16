#ifndef TEST_OPENCV_INTERVAL_H
#define TEST_OPENCV_INTERVAL_H

#include "opencv2/core/types.hpp"

struct Interval
{
    int begin, end;
    int cluster_num;
    cv::Vec3b color;
    Interval *next;

    Interval();
    Interval(int begin, int end, int cluster_num, cv::Vec3b color);
};


#endif //TEST_OPENCV_INTERVAL_H
