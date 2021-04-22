#ifndef TEST_OPENCV_POLYLINES_H
#define TEST_OPENCV_POLYLINES_H


struct Polyline
{
    int column, begin, end;
    Polyline *next;

    Polyline(int begin, int end, int column);
    int length();
};


#endif //TEST_OPENCV_POLYLINE_H
