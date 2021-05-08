#include <vector>

using namespace std;

#ifndef TEST_OPENCV_SPACEKB_H
#define TEST_OPENCV_SPACEKB_H


class SpaceKB
{
    std::vector < tuple<double, double> > points;
public:
    SpaceKB(vector< tuple<double, double> > points);
    void print_points();
    void approaching_straight_line(double &approaching_x, double &approaching_y);
};


#endif //TEST_OPENCV_SPACEKB_H
