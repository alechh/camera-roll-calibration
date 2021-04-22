#include "Polyline.h"

Polyline::Polyline(int begin, int end, int column)
{
    this->begin = begin;
    this->end = end;
    this->column = column;
    this->next = nullptr;
}

int Polyline::length()
{
    return end - begin;
}