#include "ListOfIntervalsLists.h"

/**
 * Default constructor
 */
ListOfIntervalsLists::ListOfIntervalsLists()
{
    head = nullptr;
    tail = head;
}

/**
 * Destructor
 */
ListOfIntervalsLists::~ListOfIntervalsLists()
{
    while (head != nullptr)
    {
        IntervalsList* oldHead = head;
        head = head->next;
        delete oldHead;
    }
}

/**
 * Add interval list to the list of interval lists
 * @param newIntervalList -- Pointer to the interval list
 */
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

/**
 * Length of the list
 */
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