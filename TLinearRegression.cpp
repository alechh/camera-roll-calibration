#include "TLinearRegression.h"

TLinearRegression::TLinearRegression()
{
    m_sumX = 0;
    m_sumY = 0;
    m_sumXSqr = 0;
    m_sumXY = 0;
    m_pointsCount = 0;
}

void TLinearRegression::addPoint(int y, int x)
{
    m_sumX += x;
    m_sumY += y;
    m_sumXSqr += x * x;
    m_sumXY += x * y;
    ++m_pointsCount; //it can be used as weight for lines
}

void TLinearRegression::clear()
{
    m_sumX = 0;
    m_sumY = 0;
    m_sumXSqr = 0;
    m_sumXY = 0;
    m_pointsCount = 0;
}

int TLinearRegression::get_m_pointsCount()
{
    return m_pointsCount;
}

LinearFunction TLinearRegression::calculate()
{
    double avrX = double(m_sumX) / m_pointsCount;
    double avrY = double(m_sumY) / m_pointsCount;
    double avrXSqr = double(m_sumXSqr) / m_pointsCount;
    double avrXY = double(m_sumXY) / m_pointsCount;

    // y = kx + b
    double k = (avrXY - avrX * avrY) / (avrXSqr - avrX * avrX);
    double b = (avrXSqr * avrY - avrX * avrXY) / (avrXSqr - avrX * avrX);

    // x = ky + b
//    k = 1.0 / k;
//    b = -b * k;

//    double avrX = double(m_sumX) / double(m_pointsCount);
//    double avrY = double(m_sumY) / double(m_pointsCount);
//    double avrXSqr = double(m_sumXSqr) / double(m_pointsCount);
//    double avrXY = double(m_sumXY) / double(m_pointsCount);
//
//    double k = (avrX * avrY - avrXY) / (avrX * avrX - avrXSqr);
//    double b = (avrXY - k * avrXSqr) / avrX;
//
//    // Получили y=ax+b, меняем коэфициенты, чтобы получить x=ay+b
//    k = 1.0 / k;
//    b = -b * k;

    return LinearFunction{ k, b };
}

double TLinearRegression::eps()
{
    double avrX = m_sumX / m_pointsCount;
    double avrXSqr = m_sumXSqr / m_pointsCount;
    return avrXSqr - avrX * avrX;
}

