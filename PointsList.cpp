#include <iostream>
#include "PointsList.h"

PointNode::PointNode(Point pt)
{
    this->pt = pt;
    this->next = nullptr;
}


//--------------------------------------------------------


PointsList::PointsList()
{
    this->head = nullptr;
    this->tail = nullptr;
}


PointsList::~PointsList()
{
    while (this->head)
    {
        PointNode* oldHead = head;
        head = head->next;
        delete oldHead;
    }
}

void PointsList::addPoint(Point pt)
{
    if (this->head == nullptr)
    {
        this->head = new PointNode(pt);
        this->tail = head;
        return;
    }
    this->tail->next = new PointNode(pt);
    this->tail = tail->next;
}


int PointsList::getLength()
{
    if (this->head == nullptr)
    {
        return 0;
    }
    int res = 1;
    PointNode *temp = head;
    while (temp->next != nullptr)
    {
        res++;
        temp = temp->next;
    }
    return res;
}