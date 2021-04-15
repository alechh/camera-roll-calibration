#include "Interval.h"

Interval::Interval() {};

Interval::Interval(int begin, int end, int cluster_num, cv::Vec3b color)
{
    this->begin = begin;
    this->end = end;
    this->cluster_num = cluster_num;
    this->color = color;
    this->next = nullptr;
}

Interval::Interval(int begin, int end, int cluster_num, cv::Vec3b color, Interval *next)
{
    this->begin = begin;
    this->end = end;
    this->cluster_num = cluster_num;
    this->color = color;
    this->next = next;
}

Interval::~Interval() {};