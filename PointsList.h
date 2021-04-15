#ifndef TEST_OPENCV_POINTSLIST_H
#define TEST_OPENCV_POINTSLIST_H

#include "opencv2/core/types.hpp"
using namespace cv;

struct PointNode
{
    Point pt;
    PointNode *next;

    PointNode(Point pt);
};


struct PointsList
{
    PointNode *head;

    PointsList();
    PointsList(PointNode *pt);
    void addNewPoint(Point pt);
    int getLength();
};


#endif //TEST_OPENCV_POINTSLIST_H
