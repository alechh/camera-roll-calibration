#ifndef TEST_OPENCV_LISTOFINTERVALSLISTS_H
#define TEST_OPENCV_LISTOFINTERVALSLISTS_H

#include "IntervalsList.h"

struct ListOfIntervalsLists
{
    IntervalsList *head, *tail;

    ListOfIntervalsLists();
    void addIntervalList(IntervalsList *newIntervalList);
    int get_length();
};


#endif //TEST_OPENCV_LISTOFINTERVALSLISTS_H
