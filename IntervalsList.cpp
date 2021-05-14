#include "IntervalsList.h"
#include <iostream>

/**
 * Default constructor
 */
IntervalsList::IntervalsList()
{
    head = nullptr;
    tail = nullptr;
    next = nullptr;
}

/**
 * Destructor
 */
IntervalsList::~IntervalsList()
{
    while (head != nullptr)
    {
        Interval* oldHead = head;
        head = head->next;
        delete oldHead;
    }
}

/**
 * Add interval to the list
 * @param begin -- x coordinate of the begin of the new interval
 * @param end -- x coordinate of the end of the new interval
 * @param y_coordinate -- y coordinate of the new interval
 * @param cluster_num -- number of cluster
 * @param color -- the color of the cluster that the interval belongs to
 */
void IntervalsList::addInterval(int begin, int end, int y_coordinate, int cluster_num, cv::Vec3b color)
{
    if (head == nullptr)
    {
        head = new Interval(begin, end, y_coordinate, cluster_num, color);
        tail = head;
        return;
    }
    tail->next = new Interval(begin, end, y_coordinate, cluster_num, color);
    tail = tail->next;
}

/**
 * Add interval to the list
 * @param newInterval -- Pointer to the interval
 */
void IntervalsList::addInterval(Interval *newInterval)
{
    if (head == nullptr)
    {
        head = newInterval;
        tail = head;
        return;
    }
    tail->next = newInterval;
    tail = tail->next;
}

/**
 * Length of the list
 * @return
 */
int IntervalsList::getLength()
{
    if (this->head == nullptr)
    {
        return 0;
    }
    int res = 1;
    Interval *temp = head;
    while (temp->next != nullptr)
    {
        res++;
        temp = temp->next;
    }
    return res;
}
