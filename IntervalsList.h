#ifndef TEST_OPENCV_INTERVALSLIST_H
#define TEST_OPENCV_INTERVALSLIST_H


#include "Interval.h"

struct IntervalsList
{
    Interval *head, *tail;

    IntervalsList();
    ~IntervalsList();
    IntervalsList (const IntervalsList &copy);  // конструктор копирования

    void addInterval(int begin, int end, int cluster_num, cv::Vec3b color);
    void addInterval(Interval *newInterval);
    int getLength();
    void print();
};


#endif //TEST_OPENCV_INTERVALSLIST_H
