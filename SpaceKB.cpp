#include "SpaceKB.h"
#include <iostream>
#include <tuple>
#include <utility>

// Constructor
SpaceKB::SpaceKB(set< tuple<double, double> > points): points(std::move(points)) {};

void SpaceKB::print_points()
{
    cout << "---------------------------------------" << endl;

    for (const auto & point : points)
    {
        std::cout << get<0>(point) << " ; " << get<1>(point) << endl;
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
        for (const auto & point : points)
        {
            Sx += get<0>(point);
            Sy += get<1>(point);
            Sxy += get<0>(point) * get<1>(point);
            Sxx += get<0>(point) * get<0>(point);
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