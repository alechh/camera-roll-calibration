#ifndef TEST_OPENCV_LISTOFPOLYLINES_H
#define TEST_OPENCV_LISTOFPOLYLINES_H

#include "Polyline.h"
#include <vector>
#include <opencv2/core.hpp>

struct ListOfPolylines
{
    Polyline *head, *tail;

    ListOfPolylines();
    ~ListOfPolylines();
    void addPolyline(int begin, int end, int column);
    int length();
    std::vector < std::tuple<cv::Point, cv::Point> > getPointOfLines();

};


#endif //TEST_OPENCV_LISTOFPOLYLINES_H
