#include "ListOfPolylines.h"

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