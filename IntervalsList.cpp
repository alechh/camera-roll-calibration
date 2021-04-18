#include "IntervalsList.h"
#include <iostream>

IntervalsList::IntervalsList()
{
    head = nullptr;
    tail = nullptr;
}


IntervalsList::IntervalsList(const IntervalsList *copy)
{
    if (copy->head == nullptr)
    {
        head = nullptr;
        return;
    }
    head = new Interval(copy->head->begin, copy->head->end, copy->head->cluster_num, copy->head->color);
    Interval* end = head;
    Interval* cur = copy->head->next;

    while (cur)
    {
        Interval* n = new Interval(cur->begin, cur->end, cur->cluster_num, cur->color);
        end->next = n;
        end = n;

        cur = cur->next;
    }
    tail = end;
    std::cout << "Copy" << std::endl;
    //TODO Не вызывается конструктор копирования
}


IntervalsList::~IntervalsList()
{
    while (head != nullptr)
    {
        Interval* oldHead = head;
        head = head->next;
        delete oldHead;
    }
}


void IntervalsList::addInterval(int begin, int end, int cluster_num, cv::Vec3b color)
{
    if (head == nullptr)
    {
        head = new Interval(begin, end, cluster_num, color);
        tail = head;
        return;
    }
    tail->next = new Interval(begin, end, cluster_num, color);
    tail = tail->next;
}

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

void IntervalsList::clearList()
{
    while (head != nullptr)
    {
        Interval* oldHead = head;
        head = head->next;
        delete oldHead;
    }
}