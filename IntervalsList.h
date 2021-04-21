#ifndef TEST_OPENCV_INTERVALSLIST_H
#define TEST_OPENCV_INTERVALSLIST_H


#include "Interval.h"

struct IntervalsList
{
    Interval *head, *tail;
    IntervalsList *next;

    IntervalsList();
    ~IntervalsList();

    void addInterval(int begin, int end, int y_coordinate, int cluster_num, cv::Vec3b color);
    void addInterval(Interval *newInterval);
    int getLength();
    void print();
    void clearList();
};


#endif //TEST_OPENCV_INTERVALSLIST_H
