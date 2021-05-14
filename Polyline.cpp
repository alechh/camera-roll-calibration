#include "Polyline.h"

/**
 * Constructor
 * @param begin -- the y coordinate of the top of the vertical polyline
 * @param end -- the y coordinate of the bottom of the vertical polyline
 * @param column -- x coordinate of the vertical polyline
 */
Polyline::Polyline(int begin, int end, int column)
{
    this->begin = begin;
    this->end = end;
    this->column = column;
    this->next = nullptr;
}

/**
 * Get length of the polyline
 * @return
 */
int Polyline::length()
{
    return end - begin;
}