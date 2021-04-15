#include "PointsList.h"

PointsList::PointsList()
{
    this->head = nullptr;
}

PointsList::PointsList(PointNode *pt)
{
    this->head = pt;
}

PointNode::PointNode(Point pt)
{
    this->pt = pt;
    this->next = nullptr;
}


void PointsList::addNewPoint(Point pt)
{
    if (this->head == nullptr)
    {
        this->head = new PointNode(pt);
        return;
    }
    PointNode *temp = head;
    while(temp->next != nullptr)
    {
        temp = temp->next;
    }
    temp->next = new PointNode(pt);
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