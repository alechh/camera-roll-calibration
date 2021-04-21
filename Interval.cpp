#include "Interval.h"

Interval::Interval() {};

Interval::Interval(int begin, int end, int y_coordinate, int cluster_num, cv::Vec3b color)
{
    this->begin = begin;
    this->end = end;
    this->y_coordinate = y_coordinate;
    this->cluster_num = cluster_num;
    this->color = color;
    this->next = nullptr;
}