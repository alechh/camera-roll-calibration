#ifndef TEST_OPENCV_INTERVALSLIST_H
#define TEST_OPENCV_INTERVALSLIST_H


#include "Interval.h"

struct IntervalsList
{
    Interval *head;

    IntervalsList();
    IntervalsList(Interval *interval);
    //TODO написать нормальный деструктор
//    ~IntervalsList();

    void add_interval(int begin, int end, int cluster_num, cv::Vec3b color);
    void add_interval(Interval *newInterval);
    int getLength();
    void print();
};


#endif //TEST_OPENCV_INTERVALSLIST_H
