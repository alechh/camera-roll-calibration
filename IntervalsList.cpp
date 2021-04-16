#include "IntervalsList.h"
#include <iostream>

IntervalsList::IntervalsList()
{
    this->head = nullptr;
    this->tail = nullptr;
}


IntervalsList::IntervalsList(const IntervalsList &copy):
    head(copy.head),
    tail(copy.tail)
{
    std::cout << "Copy" << std::endl;
    //TODO Не вызывается конструктор копирования
}


IntervalsList::~IntervalsList()
{
    while (this->head)
    {
        Interval* newHead = head->next;
        delete this->head;
        this->head = newHead;
    }
}


void IntervalsList::addInterval(int begin, int end, int cluster_num, cv::Vec3b color)
{
    if (this->head == nullptr)
    {
        this->head = new Interval(begin, end, cluster_num, color);
        this->tail = head;
        return;
    }
    this->tail->next = new Interval(begin, end, cluster_num, color);
    this->tail = tail->next;
}

void IntervalsList::addInterval(Interval *newInterval)
{
    if (this->head == nullptr)
    {
        this->head = newInterval;
        this->tail = head;
        return;
    }
    this->tail->next = newInterval;
    this->tail = tail->next;
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