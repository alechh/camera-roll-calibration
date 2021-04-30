#include "ListOfPolylines.h"
#include <opencv2/core/matx.hpp>


ListOfPolylines::ListOfPolylines()
{
    this->head = nullptr;
    this->tail = head;
}

void ListOfPolylines::addPolyline(int begin, int end, int column)
{
    if (head == nullptr)
    {
        head = new Polyline(begin, end, column);
        tail = head;
        return;
    }
    tail->next = new Polyline(begin, end, column);
    tail = tail->next;
}

int ListOfPolylines::length()
{
    if (head == nullptr)
    {
        return 0;
    }
    int count = 0;
    Polyline *curr = head;
    while(curr)
    {
        count++;
        curr = curr->next;
    }
    return count;
}



std::vector<std::tuple<cv::Point, cv::Point> > ListOfPolylines::getPointOfLines()
{
    std::vector< std::tuple<cv::Point, cv::Point> > result;
    if (head == nullptr)
    {
        return result;
    }

    Polyline *curr = head;
    while (curr)
    {
        cv::Point pt1, pt2;
        pt1.x = curr->column;
        pt1.y = curr->begin;

        pt2.x = curr->column;
        pt2.y = curr->end;

        result.emplace_back(pt1, pt2);

        curr = curr->next;
    }
    return result;
}

ListOfPolylines::~ListOfPolylines()
{
    while (head != nullptr)
    {
        Polyline* oldHead = head;
        head = head->next;
        delete oldHead;
    }
}




















