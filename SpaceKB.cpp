#include "SpaceKB.h"
#include <iostream>
#include <tuple>

// Constructor
SpaceKB::SpaceKB(set< tuple<double, double> > points): points(points) {};

void SpaceKB::print_points()
{
    cout << "---------------------------------------" << endl;

    for (auto i = points.begin(); i != points.end(); i++)
    {
        std::cout << get<0>(*i) << " ; " << get<1>(*i) << endl;
    }

    cout << "---------------------------------------" << endl;
}

void SpaceKB::approaching_straight_line(double &approaching_x, double &approaching_y)
{
    double a, b;
    double Sx = 0;
    double Sy = 0;
    double Sxy = 0;
    double Sxx = 0;
    int n = points.size();

    if (n > 1)
    {
        for (auto i = points.begin(); i != points.end(); i++)
        {
            Sx += get<0>(*i);
            Sy += get<1>(*i);
            Sxy += get<0>(*i) * get<1>(*i);
            Sxx += get<0>(*i) * get<0>(*i);
        }
        Sx /= n;
        Sy /= n;
        Sxy /= n;
        Sxx /= n;

        a = (Sx * Sy - Sxy) / (Sx * Sx - Sxx);
        b = (Sxy - a * Sxx) / Sx;

        a = 1.0 / a;
        b = -b * a;

        // cout << "Приближающая прямая (n=" << n << "): x = " << a << " * y + " << b << "" << endl;
        approaching_x = b;
        approaching_y = -a;
    }
}