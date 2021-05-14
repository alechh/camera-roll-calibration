#ifndef TEST_OPENCV_TLINEARREGRESSION_H
#define TEST_OPENCV_TLINEARREGRESSION_H

#include <opencv2/core.hpp>

/**
* Represents simple linear function x = k * y + b.
*/
struct LinearFunction
{
    double k;
    double b;
};

class TLinearRegression
{
    double m_sumX;
    double m_sumY;
    double m_sumXSqr;
    double m_sumXY;
    int m_pointsCount; //it can be used as weight for lines

public:
    TLinearRegression();
    void addPoint(int x, int y);
    LinearFunction calculate();
    void clear();
    double eps();
};


#endif //TEST_OPENCV_TLINEARREGRESSION_H
