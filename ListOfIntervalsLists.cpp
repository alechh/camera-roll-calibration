#include "ListOfIntervalsLists.h"

ListOfIntervalsLists::ListOfIntervalsLists()
{
    head = nullptr;
    tail = head;
}

void ListOfIntervalsLists::addIntervalList(IntervalsList *newIntervalList)
{
    if (head == nullptr)
    {
        head = newIntervalList;
        tail = head;
        return;
    }
    tail->next = newIntervalList;
    tail = tail->next;
}

int ListOfIntervalsLists::get_length()
{
    if (head == nullptr)
    {
        return 0;
    }
    int count = 0;
    IntervalsList *curr = head;
    while(curr)
    {
        count++;
        curr = curr->next;
    }
    return count;
}