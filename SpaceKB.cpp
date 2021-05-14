#include "SpaceKB.h"
#include <tuple>
#include <utility>

// Constructor
SpaceKB::SpaceKB(vector< tuple<double, double> > points): points(std::move(points)) {};

/**
 * Calculation of the approximating line through linear regression over points in KB space
 * @param approaching_x
 * @param approaching_y
 */
void SpaceKB::approaching_straight_line(double &approaching_x, double &approaching_y)
{
    int n = points.size();

    if (n > 1)
    {
        double a, b;
        double Sx = 0;
        double Sy = 0;
        double Sxy = 0;
        double Sxx = 0;

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

//        a = -a;
//        b = -b;
        //cout << "Приближающая прямая (n=" << n << "): y = " << a << " * x + " << b << "" << endl;

        // Получили y=ax+b, меняем коэфициенты, чтобы получить x=ay+b
        a = 1.0 / a;
        b = -b * a;

        //cout << "Приближающая прямая (n=" << n << "): x = " << a << " * y + " << b << "" << endl;

        approaching_x = b;
        approaching_y = a;
    }
}