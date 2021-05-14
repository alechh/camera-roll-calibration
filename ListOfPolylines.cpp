#include "ListOfPolylines.h"
#include <opencv2/core/matx.hpp>


/**
 * Default constructor
 */
ListOfPolylines::ListOfPolylines()
{
    this->head = nullptr;
    this->tail = head;
}

/**
 * Add polyline to the list
 * @param begin -- the y coordinate of the top of the vertical polyline
 * @param end -- the y coordinate of the bottom of the vertical polyline
 * @param column -- x coordinate of the vertical polyline
 */
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

/**
 * Length of the list
 * @return
 */
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

/**
 * Getting the vector of the start and end points of a segment from the list
 * @return
 */
std::vector<std::tuple<cv::Point, cv::Point> > ListOfPolylines::getPointsOfLines()
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

/**
 * Destructor
 */
ListOfPolylines::~ListOfPolylines()
{
    while (head != nullptr)
    {
        Polyline* oldHead = head;
        head = head->next;
        delete oldHead;
    }
}




















