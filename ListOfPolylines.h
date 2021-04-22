#ifndef TEST_OPENCV_LISTOFPOLYLINES_H
#define TEST_OPENCV_LISTOFPOLYLINES_H

#include "Polyline.h"

struct ListOfPolylines
{
    Polyline *head, *tail;

    ListOfPolylines();
    void addPolyline(int begin, int end, int column);
    int length();
};


#endif //TEST_OPENCV_LISTOFPOLYLINES_H
