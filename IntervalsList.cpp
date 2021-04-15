#include "IntervalsList.h"
#include <iostream>

IntervalsList::IntervalsList()
{
    this->head = nullptr;
}

IntervalsList::IntervalsList(Interval *interval)
{
    this->head = interval;
}


//IntervalsList::~IntervalsList()
//{
//    if (this->head == nullptr)
//    {
//        return;
//    }
//    Interval *temp1, *temp2;
//    temp1 = head;
//    while (temp1->next != nullptr)
//    {
//        temp2 = temp1;
//        temp1 = temp1->next;
//        delete temp2;
//    }
//    delete temp1;
//}


void IntervalsList::add_interval(int begin, int end, int cluster_num, cv::Vec3b color)
{
    if (this->head == nullptr)
    {
        this->head = new Interval(begin, end, cluster_num, color);
        return;
    }
    Interval *temp = head;
    while (temp->next != nullptr)
    {
        temp = temp->next;
    }
    temp->next = new Interval(begin, end, cluster_num, color);
}

void IntervalsList::add_interval(Interval *newInterval)
{
    if (this->head == nullptr)
    {
        this->head = newInterval;
        return;
    }
    Interval *temp = head;
    while (temp->next != nullptr)
    {
        temp = temp->next;
    }
    temp->next = newInterval;
}


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

void IntervalsList::print()
{
    if (this->head == nullptr)
    {
        return;
    }
    Interval *temp = head;
    while (temp != nullptr)
    {
        std::cout << "[" << temp->begin << "," << temp->end << "]";
        if (temp->next != nullptr)
        {
            std::cout << " -> ";
        }
        temp = temp->next;
    }
    std::cout << std::endl;
}