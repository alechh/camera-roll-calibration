#include "TLinearRegression.h"

/**
 * Default constructor
 */
TLinearRegression::TLinearRegression()
{
    m_sumX = 0;
    m_sumY = 0;
    m_sumXSqr = 0;
    m_sumXY = 0;
    m_pointsCount = 0;
}

/**
 * Add point to the regression
 * @param y -- y coordinate of the point
 * @param x -- x coordinate of the point
 */
void TLinearRegression::addPoint(int y, int x)
{
    m_sumX += x;
    m_sumY += y;
    m_sumXSqr += x * x;
    m_sumXY += x * y;
    ++m_pointsCount; //it can be used as weight for lines
}

/**
 * Reset the regression
 */
void TLinearRegression::clear()
{
    m_sumX = 0;
    m_sumY = 0;
    m_sumXSqr = 0;
    m_sumXY = 0;
    m_pointsCount = 0;
}


/**
 * Calculate the linear function x = k * y + b via the regression
 * @return
 */
LinearFunction TLinearRegression::calculate()
{
    double avrX = double(m_sumX) / m_pointsCount;
    double avrY = double(m_sumY) / m_pointsCount;
    double avrXSqr = double(m_sumXSqr) / m_pointsCount;
    double avrXY = double(m_sumXY) / m_pointsCount;

    // x = k * y + b
    double k = (avrXY - avrX * avrY) / (avrXSqr - avrX * avrX);
    double b = (avrXSqr * avrY - avrX * avrXY) / (avrXSqr - avrX * avrX);


    return LinearFunction{ k, b };
}

double TLinearRegression::eps()
{
    double avrX = m_sumX / m_pointsCount;
    double avrXSqr = m_sumXSqr / m_pointsCount;
    return avrXSqr - avrX * avrX;
}

