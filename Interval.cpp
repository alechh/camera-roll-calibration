#include "Interval.h"
#include <iostream>

Interval::Interval() {};

Interval::Interval(int begin, int end, int cluster_num, cv::Vec3b color)
{
    this->begin = begin;
    this->end = end;
    this->cluster_num = cluster_num;
    this->color = color;
    this->next = nullptr;
}