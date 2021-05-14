#include "Interval.h"

/**
 * Default constructor
 */
Interval::Interval() {};

/**
 * Consturcor
 * @param begin -- x coordinate of the begin of the new interval
 * @param end -- x coordinate of the end of the new interval
 * @param y_coordinate -- y coordinate of the new interval
 * @param cluster_num -- number of cluster
 * @param color -- the color of the cluster that the interval belongs to
 */
Interval::Interval(int begin, int end, int y_coordinate, int cluster_num, cv::Vec3b color)
{
    this->begin = begin;
    this->end = end;
    this->y_coordinate = y_coordinate;
    this->cluster_num = cluster_num;
    this->color = color;
    this->next = nullptr;
    this->added = false;
}