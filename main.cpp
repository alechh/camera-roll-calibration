#include <iostream>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "SpaceKB.h"
#include "IntervalsList.h"
#include "Interval.h"
#include "ListOfIntervalsLists.h"
#include "ListOfPolylines.h"
#include "TLinearRegression.h"

using namespace cv;
using namespace std;

/**
 * Calculation of points of a straight line along rho and theta
 * @param rho -- radial coordinate
 * @param theta -- angular coordinate
 * @param pt1 -- Point 1
 * @param pt2 -- Point 2
 */
void calculatingPoints(double rho, double theta, Point &pt1, Point &pt2)
{
    double a, b, x0, y0;
    a = cos(theta);
    b = sin(theta);

    x0 = a * rho;
    y0 = b * rho;

    pt1.x = cvRound(x0 - 1000 * b);
    pt1.y = cvRound(y0 + 1000 * a);
    pt2.x = cvRound(x0 + 1000 * b);
    pt2.y = cvRound(y0 - 1000 * a);
}

/**
 * Making SpaceKB to find the approximating line through linear regression
 * @param vertical_lines -- vector of points through which the lines pass
 */
void makeSpaceKB(double &result_x, double &result_y, vector <tuple<Point, Point>> vertical_lines)
{
    vector< tuple<double, double> > coefficientsKB;  // Коэффициенты b и k прямых x = ky+b для построения пространства Kb

    for (auto & vertical_line : vertical_lines)
    {
        Point pt1, pt2;

        pt1 = get<0>(vertical_line);
        pt2 = get<1>(vertical_line);

        // x = k * y + b
        // Беру k со знаком -, потому что в OpenCV система координат инвертирована относительно оси OY
        double k = - double((pt2.x - pt1.x)) / (pt2.y - pt1.y);
        double b = pt1.x - pt1.y * double(pt2.x - pt1.x) / (pt2.y - pt1.y);

        coefficientsKB.emplace_back(make_tuple(b, k));
    }

    if (coefficientsKB.begin() != coefficientsKB.end())  // если нашлось хотя бы 2 прямые
    {
        double approaching_x = -1;
        double approaching_y = -1;

        SpaceKB spaceKb(coefficientsKB);

        spaceKb.approaching_straight_line(approaching_x, approaching_y);  // вычисление координат точки пересечения прямых

        if (approaching_x != -1 && approaching_y != -1)
        {
            result_x = approaching_x;
            result_y = approaching_y;
        }
    }
}

/**
 * Select the vertical lines from the all lines
 * @param lines
 * @param delta
 * @return
 */
vector <tuple<Point, Point>> selectionOfVerticalLines(vector<Vec2f> lines, int delta = 300)
{
    vector <tuple<Point, Point>> vertical_lines;  // множество пар точек, через которые проходят вертикальные прямые

    for (auto & line : lines)
    {
        double rho, theta;
        Point pt1, pt2;  // 2 точки, через которые проходит прямая

        rho = line[0];
        theta = line[1];

        calculatingPoints(rho, theta, pt1, pt2);  // вычисление двух точек прямой (pt1, pt2)

        if (abs(pt1.x - pt2.x) < delta)  // если прямая подозрительна на вертикальную
        {
            vertical_lines.emplace_back(pt1, pt2);
        }
    }
    return vertical_lines;
}

/**
 * Draw lines
 * @param src -- Input image
 * @param lines -- vector of pairs of points of a straight line
 */
void drawLines(Mat &src, vector <tuple<Point, Point>> lines)
{
    for (auto & line : lines)
    {
        Point pt1, pt2;
        pt1 = get<0>(line);
        pt2 = get<1>(line);

        cv::line(src, pt1, pt2, CV_RGB(0,255,0), 1, LINE_AA);  // отрисовка прямой
    }
}

 /**
  * Search for vertical straight lines on video using the Hough method
  * @param src -- Input image
  * @return vector of rho and theta pairs
  */
vector<Vec2f> findLinesHough(Mat &src)
{
    Mat src_gray, src_canny;
    vector<Vec2f> lines;  // прямые, найденные на изображении

    // Median blur-----------------------
//     int n = 3;
//     medianBlur(src, src, n);

    //cvtColor(src, src_gray, COLOR_BGR2GRAY);  // Подготовка изображения для метода Хафа поиска прямых
    Canny(src, src_canny, 50, 200);  // Подготовка изображения для метода Хафа поиска прямых

    HoughLines(src_canny, lines, 1, CV_PI / 180, 50);

    src_gray.release();
    src_canny.release();
    return lines;
}

/**
 * Draw vertical line
 * @param src -- Input image
 * @param x -- x coordinate of the line
 */
void drawVerticalLine(Mat &src, double x)
{
    Point pt1, pt2;

    pt1.x = x;
    pt1.y = 0;
    pt2.x = x;
    pt2.y = src.rows;

    if (pt1.x != 0 && pt2.x != 0)
    {
        line(src, pt1, pt2, CV_RGB(80, 222, 24), 6, CV_AA);
    }
}

/**
 * Draw x coordinate of the point on the image
 * @param src -- Input image
 * @param x -- x coordinate
 */
void drawXOnImage(Mat &src, double x)
{
    String text = to_string(x);
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 2;
    int thickness = 3;
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    Point textOrg(15,50);  // Position of the text

    putText(src, text, textOrg, fontFace, fontScale,
            Scalar(0, 0, 0), thickness, 8);
}

/**
 * Simple buble sort for array of double
 * @param values -- array of double
 * @param size -- size of the array
 */
void bubbleSort(double *values, int size)
{
    // сперва нужно перенести все NaN в конец массива
    int index_of_last_not_nan;

    // получение индекса последнего числа != nan
    for (int i = size - 1; i >= 0; i--)
    {
        if (!std::isnan(values[i]))
        {
            index_of_last_not_nan = i;
            break;
        }
    }

    // перемещение nan в конец массива
    for (int i = 0; i < size; i++)
    {
        if (std::isnan(values[i]))
        {
            std::swap(values[i], values[index_of_last_not_nan]);

            for (int j = size - 1; j >= 0; j--)
            {
                if (!std::isnan(values[j]))
                {
                    index_of_last_not_nan = j;
                    break;
                }
            }
        }
    }

    // сортировка
    for (size_t i = 0; i + 1 < size; i++)
    {
        for (size_t j = 0; j + 1 < size - i; j++)
        {
            if (values[j + 1] < values[j] && !std::isnan(values[j + 1]) && !std::isnan(values[j]))
            {
                std::swap(values[j], values[j + 1]);
            }
        }
    }
}

/**
 * The median value in the array
 * @param valuesForMedianFilter -- array of values for median filter
 * @param NUMBER_OF_MEDIAN_VALUES -- a number indicating the frequency of the median filter
 * @return -- median value
 */
double medianFilter(double *valuesForMedianFilter, const int NUMBER_OF_MEDIAN_VALUES)
{
    int indexOfResult = (NUMBER_OF_MEDIAN_VALUES - 1) / 2;

    bubbleSort(valuesForMedianFilter, NUMBER_OF_MEDIAN_VALUES);

    return valuesForMedianFilter[indexOfResult];
}


/**
 * Function of finding straight vertical lines using Hough method
 * @tparam T
 * @param path -- path to the video file. 0 means that the video will be read from the webcam.
 * @param resize -- image resizing factor.
 */
template <class T>
void selectingLinesUsingHoughMethod(T path, double resize = 1)
{
    VideoCapture capture(path);
    if (!capture.isOpened())
    {
        cerr<<"Error"<<endl;
        return;
    }

    Mat src;

    int n = 1;  // Счетчик для медианного фильтра
    const int NUMBER_OF_MEDIAN_VALUES = 20;  // Раз во сколько кадров проводим медианный фильтр
    double valuesForMedianFilter[NUMBER_OF_MEDIAN_VALUES - 1];  // Массив значений, который будет сортироваться для медианного фильтра
    double prevResult_x = 0;  // Сохранение предыдущего значения, чтобы выводить на экран

    while (true)
    {
        capture >> src;

        vector<Vec2f> lines = findLinesHough(src);  // нахождение прямых линий

        vector <tuple<Point, Point>> vertical_lines = selectionOfVerticalLines(lines);  // выбор только вертикальных линий

        drawLines(src, vertical_lines);  // отрисовка прямых линий

        double result_x = 0;  // Вычисленная координата x точки схода прямых
        double result_y = 0;

        makeSpaceKB(result_x, result_y, vertical_lines);  // построение пространства Kb, чтобы найти приближающую прямую через
                                                  // линейную регрессию. Далее обратным отображением находим точку схода прямых

        if (n % NUMBER_OF_MEDIAN_VALUES == 0)  // Если нужно провести медианный фильтр
        {
            prevResult_x = medianFilter(valuesForMedianFilter, NUMBER_OF_MEDIAN_VALUES);
        }
        else
        {
            valuesForMedianFilter[n - 1] = result_x;
        }

        drawVerticalLine(src, prevResult_x);

        if (resize != 1)
        {
            cv::resize(src, src, cv::Size(), resize, resize);
        }

        imshow("Result", src);

        if (n % NUMBER_OF_MEDIAN_VALUES == 0)
        {
            n = 1;
        }
        else
        {
            n++;
        }

        int k = waitKey(25);
        if (k == 27)
        {
            break;
        }
    }
}

/**
 * Selection of vertical continuous segments, depending on their length, draw them in dst
 * @param src -- Input image
 * @param dst -- Output image
 * @param delta -- The coefficient by which we glue close segments together
 */
void makePolylines(Mat &src, Mat &dst, int delta = 10, int x_roi = 0, int width_roi = 0)
{
    auto listOfPolylines = new ListOfPolylines;

    // заполняем listOfPolylines вертикальными отрезками
    for (int i = 0; i < src.cols; i++)
    {
        int begin = 0;
        int end = 0;

        for (int j = 1; j < src.rows; j++)
        {

            cv::Vec3b lastColor = src.at<Vec3b>(j - 1, i);
            cv::Vec3b newColor = src.at<Vec3b>(j, i);

            if (newColor != lastColor)
            {
                if (newColor == Vec3b(0, 0, 0)) // встретили черный цвет, значит это начало нового отрезка
                {
                    if (j - end < delta)
                    {
                        // если есть 2 отрезка в одной колонке, которые очень близко, но разрвны с разрывом длиной delta
                        // тогда мы их соединяем
                        // end -- конец предыдущего найденного отрезка

                        // также добавим фильтр roi (будем склеивать только центральные отрезки)
                        if (width_roi != 0)
                        {
                            if (x_roi <= i && i <= x_roi + width_roi)
                            {
                                begin = end + 1;
                            }
                            else
                            {
                                begin = j;
                            }
                        }
                        else
                        {
                            begin = end + 1;
                        }

                    }
                    else
                    {
                        begin = j;
                    }
                }
                else if (newColor == Vec3b(255, 255, 255))
                {
                    // встретили белый цвет, значит это конец отрезка
                    end = j - 1;
                    listOfPolylines->addPolyline(begin, end, i);
                }
            }
        }
    }

    // выводим вериткальные отрезки на изображение
    Polyline *currPolyline = listOfPolylines->head;
    while (currPolyline)
    {
        if (50 <= currPolyline->length() && currPolyline->length() <= src.rows)
        {
            Point pt1, pt2;

            pt1.x = currPolyline->column;
            pt1.y = currPolyline->begin;

            pt2.x = currPolyline->column;
            pt2.y = currPolyline->end;

            line(dst, pt1, pt2, Scalar(0, 0, 0), 1);
        }
        currPolyline = currPolyline->next;
    }

    // освобождаем память
    delete listOfPolylines;
}

/**
 *  Gluing the borders of clusters that match the same color
 * @param src -- Input image
 * @param dst -- Output image
 */
void vectorisation(Mat &src, Mat &dst)
{
    auto *listOfIntervalsLists = new ListOfIntervalsLists;
    int begin = 0;
    int end;

    // заполняем список списков интервалов для первой строки
    for (int j = 1; j < src.cols; j++)
    {
        if (src.at<Vec3b>(0, j) != src.at<Vec3b>(0, j - 1))
        {
            end = j - 1;
            auto *intervalList = new IntervalsList;
            intervalList->addInterval(begin, end, 0, src.at<Vec3b>(0, j - 1));
            begin = j;

            listOfIntervalsLists->addIntervalList(intervalList);
        }
    }
    // последний интервал в первой строке
    auto *intervalList = new IntervalsList;
    intervalList->addInterval(begin, src.cols - 1, 0, src.at<Vec3b>(0, src.cols - 1));
    listOfIntervalsLists->addIntervalList(intervalList);

    // заполняем списки интервалов для остальных строк
    for (int i = 1; i < src.rows; i++)
    {
        // заполним первый интервал мусорными значениями (не добавляем его в список)
        auto *currInterval = new Interval(-1, -1, -1, src.at<Vec3b>(0, 0));

        IntervalsList *currIntervalList = listOfIntervalsLists->head;

        int prevIntervalEnd = currIntervalList->tail->end;  // эта переменная нужна для своевременного сдвига текущего списка интервалов

        begin = 0;

        for (int j = 1; j < src.cols; j++)
        {
            if (src.at<Vec3b>(i, j) != src.at<Vec3b>(i, j - 1))  // если нашли границу интервала
            {
                cv::Vec3b color = src.at<Vec3b>(i, j - 1);  // цвет найденного интервала

                // удаление интервалов, которые мы не добавили (чтобы память не утекала)
                if (!currInterval->added)
                {
                    delete currInterval;
                }

                end = j - 1;
                currInterval = new Interval(begin, end, i, color);
                begin = j;

                // сравниваем найденный интервал с интервалом из списка интервалов предыдущей строки
                // (они точно пересекаются, потому что мы вовремя двигаем указатель на список интервалов)
                if (color == currIntervalList->tail->color)  // если интервалы относятся к одному кластеру (один цвет)
                {
                    currInterval->added = true;
                    currIntervalList->addInterval(currInterval);  // сохраняем интервал в списке
                }
            }

            // сдвигаем указатель на список интервалов
            while (prevIntervalEnd <= currInterval->end && currIntervalList->next != nullptr)
            {
                currIntervalList = currIntervalList->next;
                prevIntervalEnd = currIntervalList->tail->end;
            }
        }

        if (!currInterval->added)
        {
            delete currInterval;
        }

        // добавление последнего интервала строки
        currInterval = new Interval(begin, src.cols - 1, i, src.at<Vec3b>(i, src.cols - 1));
        if (currInterval->color == currIntervalList->tail->color)
        {
            currIntervalList->addInterval(currInterval);
        }
        else
        {
            delete currInterval;
        }
    }

    // нарисуем границы кластеров
    Mat result_of_vectorisation(src.rows, src.cols, src.type(), Scalar(255, 255, 255));
    auto currIntervalList = listOfIntervalsLists->head;

    while (currIntervalList != nullptr)
    {
        auto currInterval = currIntervalList->head;

        while (currInterval != nullptr)
        {
            Point pt1, pt2;
            pt1.x = currInterval->begin;
            pt1.y = currInterval->y_coordinate;
            pt2.x = currInterval->end;
            pt2.y = currInterval->y_coordinate;

            circle(result_of_vectorisation, pt1, 1, Scalar(0, 0, 0), 1);
            circle(result_of_vectorisation, pt2, 1, Scalar(0, 0, 0), 1);

            currInterval = currInterval->next;
        }
        currIntervalList = currIntervalList->next;
    }
    dst = result_of_vectorisation;

    // освобождаем память
    result_of_vectorisation.release();
    delete listOfIntervalsLists;
}

/**
 * Paint each pixel in its own color depending on the angle of the gradient
 * @param grad_x -- Array for x gradient
 * @param grad_y -- Array for y gradient
 * @param dst -- Output image
 */
void clustering(Mat& grad_x, Mat& grad_y, Mat& dst)
{
    Mat angle(grad_x.rows, grad_x.cols, CV_64FC4);
    phase(grad_x, grad_y, angle);  // вычисление углов градиента в каждой точке

    MatIterator_<Vec3b> it, end;
    int i = 0;
    int j = 0;

    for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; ++it)
    {
        float s = angle.at<float>(i, j);

        if (- CV_PI / 4 <= s && s <= CV_PI / 4)
        {
            // red
            (*it)[0] = 0;
            (*it)[1] = 0;
            (*it)[2] = 255;
        }
        else if (CV_PI / 4 < s && s <= 3.0 * CV_PI / 4)
        {
            // blue
            (*it)[0] = 255;
            (*it)[1] = 0;
            (*it)[2] = 0;
        }
        else if (3.0 * CV_PI / 4 < s && s <= 5.0 * CV_PI / 4)
        {
            // green
            (*it)[0] = 0;
            (*it)[1] = 255;
            (*it)[2] = 0;
        }
        else if (5.0 * CV_PI / 3 < s && s <= 7.0 * CV_PI / 4)
        {
            // yellow
            (*it)[0] = 0;
            (*it)[1] = 255;
            (*it)[2] = 255;
        }

        j++;
        if (j == angle.cols)
        {
            i++;
            j = 0;
        }
    }

    // освобождаем память
    angle.release();
}

/**
 * Calculating the gradient angles
 * @param src -- Input image
 * @param grad_x -- Output for x gradient
 * @param grad_y -- Output for y gradient
 */
void simpleSobel(Mat &src, Mat &grad_x, Mat &grad_y)
{
    Mat src_gauss, src_gray;

    GaussianBlur(src, src_gauss, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cvtColor(src_gauss, src_gray, COLOR_BGR2GRAY);

    Sobel(src_gray, grad_x, CV_32F, 1, 0);
    Sobel(src_gray, grad_y, CV_32F, 0, 1);

    // освобождаем память
    src_gauss.release();
    src_gray.release();
}

/**
 * Discarding lines that are outside the Region Of Interest
 * @param lines -- vector of pairs of points of vertical lines
 * @param x_roi -- left border of the roi
 * @param width_roi -- width of the roi
 */
void roiForVerticalLines(vector< tuple<Point, Point> > &lines, int x_roi, int width_roi)
{
    auto i = lines.begin();

    while (i != lines.end())
    {
        Point pt1 = get<0>(*i);
        Point pt2 = get<1>(*i);

        if (pt1.x < x_roi || pt1.x > x_roi + width_roi)
        {
            i = lines.erase(i);
        }
        else
        {
            i++;
        }
    }
}

/**
 * A filter that allows to determine whether the same number of straight lines were found
 * to the left and right of the middle of the image
 * @param vertical_lines -- vector of pairs of points of a straight line
 * @param src_center -- x coordinate of the center of the image
 * @param delta -- Acceptable value for the deviation of the number of left segments from the number of right segments
 * @return
 */
bool quantitativeFilter(vector < tuple<Point, Point> > vertical_lines, int src_center, double delta = 2)
{
    int countLeft = 0;

    // считаем количество прямых левее центра изображения
    for (auto & line : vertical_lines)
    {
        Point pt1 = get<0>(line);
        if (pt1.x <= src_center)
        {
            countLeft++;
        }
    }

    if (countLeft == 0 || vertical_lines.size() - countLeft == 0)
    {
        return false;
    }

    // если слева и справа прямых почти поровну (количество отличается на delta)
    if (countLeft / (vertical_lines.size() - countLeft) < delta)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/**
 * Selecting the horizon line manually. Important: for each video, the result must be different
 * @param src -- Input image
 * @return tuple of the points of the horizon
 */
tuple<Point, Point> manuallySelectingHorizonLine(Mat src)
{
    Point pt1, pt2;

    // for video PATH_road3
    pt1.x = 0;
    pt1.y = src.rows / 2 + 12;

    pt2.x = src.cols - 1;
    pt2.y = src.rows / 2 + 78;

    return make_tuple(pt1, pt2);
}

/**
 * Get points of the accurate horizon line
 * @param y -- y coordinate of the horizon line
 * @param width -- width of the image
 * @return tuple of points of the line
 */
tuple<Point, Point>  getAccurateHorizonLine(int y, int width)
{
    Point pt1, pt2;

    pt1.x = 0;
    pt1.y = y;
    pt2.x = width - 1;
    pt2.y = y;

    return make_tuple(pt1, pt2);
}

/**
 * Get points of the accurate vertical line
 * @param x -- x coordinate of the vertical line
 * @param height -- height of the image
 * @return tuple of points of the line
 */
tuple<Point, Point> getAccurateVerticalLine(int x, int height)
{
    Point pt1, pt2;

    pt1.x = x;
    pt1.y = 0;

    pt2.x = x;
    pt2.y = height - 1;

    return make_tuple(pt1, pt2);
}

/**
 * Calculating the vanishing point of straight road markings
 * @param roadMarkings -- vector of straight line point pairs of road markings
 * @return vanishing point
 */
Point findVanishingPointLane( vector< tuple<Point, Point> > roadMarkings)
{
    vector< tuple<double, double> > coefficientsKB;
    double result_x, result_y;

    for (auto & line : roadMarkings)
    {
        Point pt1, pt2;
        pt1 = get<0>(line);
        pt2 = get<1>(line);

        // x = k * y + b
        double k = - double((pt2.x - pt1.x)) / (pt2.y - pt1.y);
        double b = pt1.x - pt1.y * double(pt2.x - pt1.x) / (pt2.y - pt1.y);

        coefficientsKB.emplace_back(make_tuple(b, k));
    }

    if (coefficientsKB.begin() != coefficientsKB.end())  // если нашлось хотя бы 2 прямые
    {
        SpaceKB spaceKb(coefficientsKB);
        spaceKb.approaching_straight_line(result_x, result_y);  // вычисление координат точки пересечения прямых
    }

    Point van_point_lane;

    // можно пользоваться заранее вычисленной точкой, а можно вычислять ее онлайн
//     van_point_lane.x = result_x;
//     van_point_lane.y = result_y;

     // for video PATH_road3
    van_point_lane.x = 586;
    van_point_lane.y = 428;

    return van_point_lane;
}

/**
 * Calculating road marking lines from an image
 * @param src -- Input image
 * @return vector of straight line point pairs of road markings
 */
vector< tuple<Point, Point> > findRoadMarkingLines(Mat &src)
{
    Mat src_canny;

    Mat src_hsv = Mat(src.cols, src.rows, 8, 3);
    vector<Mat> splitedHsv = vector<Mat>();
    cvtColor(src, src_hsv, CV_RGB2HSV);
    split(src_hsv, splitedHsv);

    int sensivity = 120;

    int S_WHITE_MIN = 0;
    int V_WHITE_MIN = 255 - sensivity;

    int S_WHITE_MAX = sensivity;
    int V_WHITE_MAX = 255;


    for (int y = 0; y < src_hsv.cols; y++)
    {
        for (int x = 0; x < src_hsv.rows; x++)
        {
            // получаем HSV-компоненты пикселя
            int H = static_cast<int>(splitedHsv[0].at<uchar>(x, y));        // Тон
            int S = static_cast<int>(splitedHsv[1].at<uchar>(x, y));        // Интенсивность
            int V = static_cast<int>(splitedHsv[2].at<uchar>(x, y));        // Яркость

            if (!(S_WHITE_MIN <= S && S <= S_WHITE_MAX && V_WHITE_MIN <= V && V <= V_WHITE_MAX))
            {
                src_hsv.at<Vec3b>(x, y)[0] = 0;
                src_hsv.at<Vec3b>(x, y)[1] = 0;
                src_hsv.at<Vec3b>(x, y)[2] = 0;
            }
        }
    }

    Canny(src_hsv, src_canny, 200, 360);

    vector<Vec4f> lines;
    HoughLinesP(src_canny, lines, 1, CV_PI / 180, 150, 5, 8);

    vector< tuple<Point, Point> > roadMarkings; // здесь будем хранить точки прямых левее и правее от центра

    for (auto & line : lines)
    {
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];

        if (x1 > x2)
        {
            // делаем, чтобы крайняя левая точка прямой была (x1,y1) для удобства
            int temp_x1 = x1;
            int temp_y1 = y1;
            x1 = x2;
            y1 = y2;
            x2 = temp_x1;
            y2 = temp_y1;
        }

        if (x2 != x1) // если прямая не вертикальная
        {
            const double MIN_SLOPE = 0.3;
            const double MAX_SLOPE = 0.7;

            double slope = double(y2 - y1) / (x2 - x1);
            int epsilon = 20;

            if (MIN_SLOPE <= abs(slope) && abs(slope) <= MAX_SLOPE && y2 >= src.rows / 2 && y1 >= src.rows / 2 && abs(y2 - y1) >= epsilon)
            {
                // если нашлась прямая с нужным наклоном, в нижней половине изображения и достаточная по длине
                roadMarkings.emplace_back(make_tuple(Point(x1, y1), Point(x2, y2)));
            }
        }
    }

    // draw lines
//    for (auto & road_line : roadMarkings)
//    {
//        line(src, get<0>(road_line), get<1>(road_line), Scalar(255, 255, 255), 2);
//    }

    src_hsv.release();
    src_canny.release();

    return roadMarkings;
}

/**
 * Distance from a point to a straight line
 * @param pnt -- Point
 * @param f -- Linear function x = k * y + b
 * @return  distance
 */
double distToLine(Point pnt, LinearFunction f)
{
    // distance from point to line x - f.k * y - f.c = 0
    return abs(double(pnt.x) - f.k * double(pnt.y) - f.b) / (sqrt(1 + f.k * f.k));
}

/**
 * Calculating linear regressions based on contour points
 * @param m_points -- Points of the contour
 * @return vector of TLinearRegression
 */
vector<TLinearRegression> calcRegressions(vector<Point> m_points)
{
    vector<TLinearRegression> regressions;

    if (m_points.size() < 2)
    {
        return regressions;
    }

    TLinearRegression regression;

    //add points 0-2 to regression
    for (size_t i = 0; i < 3; i++)
    {
        regression.addPoint(m_points[i].x, m_points[i].y);
    }

    LinearFunction f = regression.calculate();

    double c_min_dist_threshold = 2;

    for (size_t i = 3; i < m_points.size(); ++i)
    {
        Point pnt = m_points[i];

        if (distToLine(pnt, f) < c_min_dist_threshold) //also try to use regression.eps() to appreciate curvature of added points sequence
        {
            regression.addPoint(m_points[i].x, m_points[i].y);
        }
        else
        {
            regressions.emplace_back(regression);
            regression.clear();

            while (i < m_points.size() - 2)
            {
                if (m_points[i].x == m_points[i + 1].x && m_points[i + 1].x == m_points[i + 2].x)
                {
                    // если у следующих 3х точек одна абсцисса
                    i++;
                }
                else
                {
                    regression.addPoint(m_points[i].x, m_points[i].y);
                    regression.addPoint(m_points[i + 1].x, m_points[i + 1].y);
                    regression.addPoint(m_points[i + 2].x, m_points[i + 2].y);
                    i += 2;
                    break;
                }
            }
        }
        f = regression.calculate();
    }

    if (regressions.size() == 0)
    {
        // если все точки контура лежат на одной прямой, то regression так и не добавится в regressions
        regressions.emplace_back(regression);
    }
    return regressions;
}

/**
 * Getting two points of a straight line x = k * y + b through the coefficients k,b
 * @param k -- the angle of inclination is straight
 * @param b -- offset
 * @return a tuple of straight points
 */
tuple<Point, Point> getPointsOfTheLine(double k, double b)
{
    // x = k * y + b.
    Point pt1, pt2;

    pt1.y = 0;
    pt1.x = k * pt1.y + b;

    pt2.y = 720;
    pt2.x = k * pt2.y + b;

    return make_tuple(pt1, pt2);
}

/**
 * Obtaining a vector of linear functions x = k * y + b via linear regression over contours
 * @param contours -- vector of contours
 * @return vector of LinearFunctions x = k * y + b
 */
vector <LinearFunction> linearRegressionForContours(vector< vector<Point> > contours)
{
    vector <LinearFunction> linearFunctions; // вектор коэффициентов {k, b} линейных функций x = k * y + b

    // линейная регрессия по всем контурам
    for (size_t i = 0; i < contours.size(); i++)
    {
        // считаем регрессии от контура
        vector <TLinearRegression> regressions = calcRegressions(contours[i]);

        if (regressions.size() == 0)
        {
            continue;
        }

        // вычисление линейных функций x = k * y + b по полученным регрессиям
        for (size_t j = 0; j < regressions.size(); j++)
        {
            linearFunctions.template emplace_back(regressions[j].calculate());
        }
    }

    return linearFunctions;
}

/**
 * Getting a vector of pairs of straight points from a vector of linear functions
 * @param linearFunctions -- vector of Linear Functions x = k * y + b
 * @return vector of tuple of points of the straight lines
 */
vector < tuple<Point, Point> > findVectorOfPointsOfVerticalLines(vector <LinearFunction> linearFunctions)
{
    vector < tuple<Point, Point> > vertical_lines;

    for (int i = 0; i < linearFunctions.size(); i++)
    {
        double delta = 0.05; // чтобы определить вертикальные прямые через угол наклона

        if (!std::isnan(linearFunctions[i].k) && !std::isnan(linearFunctions[i].b) && abs(linearFunctions[i].k) < delta)
        {
            Point pt1, pt2;
            tuple<Point, Point> points = getPointsOfTheLine(linearFunctions[i].k, linearFunctions[i].b);

            pt1 = get<0>(points);
            pt2 = get<1>(points);

            vertical_lines.template emplace_back(make_tuple(pt1, pt2));
        }
    }

    return vertical_lines;
}

/**
 * Get angle between two straight lines
 * @param pt11 -- Point of the first line l1
 * @param pt12 -- Point of the first line l1
 * @param pt21 -- Point of the second line l2
 * @param pt22 -- Point of the second line l2
 * @return angle in degrees between l1 and l2
 */
double angleBetweenStraightLines(Point pt11, Point pt12, Point pt21, Point pt22)
{
    // y = k * x + b
    double k1, b1, k2, b2;

    if (pt12.x - pt11.x == 0)
    {
        // если прямая вертикальная
        k1 = pt11.x;
        b1 = 0;
    }
    else
    {
        k1 = double(pt12.y - pt11.y) / (pt12.x - pt11.x);
        b1 = - pt11.x * double(pt12.y - pt11.y) / (pt12.x - pt11.x) + pt11.y;
    }

    if (pt22.x - pt21.x == 0)
    {
        // если прямая вертикальная
        k2 = pt21.x;
        b2 = 0;
    }
    else
    {
        k2 = double(pt22.y - pt21.y) / (pt22.x - pt21.x);
        b2 = - pt21.x * double(pt22.y - pt21.y) / (pt22.x - pt21.x) + pt21.y;
    }


    double angle_radian =  abs(atan((k2 - k1) / (1 + k2 * k1)));
    double angle_degree = angle_radian * (180.0 / CV_PI);

    return angle_degree;
}


template <class T>
void selectingLinesUsingGradient(T path, double resize = 1)
{
    VideoCapture capture(path);
    if (!capture.isOpened())
    {
        cerr<<"Error"<<endl;
        return;
    }

    Mat src, src_vectorization;
    Mat grad_x, grad_y;

//     VideoWriter outputVideo;
//     Size S = Size((int) capture.get(CAP_PROP_FRAME_WIDTH), (int) capture.get(CAP_PROP_FRAME_HEIGHT));
//     int ex = static_cast<int>(capture.get(CAP_PROP_FOURCC));
//     outputVideo.open("../result.mp4", ex, capture.get(CAP_PROP_FPS), S, true);

    int medianCount = 1;  // Счетчик для медианного фильтра
    const int NUMBER_OF_MEDIAN_VALUES = 4;  // Раз во сколько кадров проводим медианный фильтр
    double valuesForMedianFilterX[NUMBER_OF_MEDIAN_VALUES - 1]; // массив значений координат x, который будет сортироваться для медианного фильтра
    double valuesForMedianFilterY[NUMBER_OF_MEDIAN_VALUES - 1]; // массив значений координат y, чтобы после медианного фильтра восстановить точку van_point_verticals
    double medianResult_x = 0;  // медианное значение для точки схода

    double result_x = 0; // координата x точки схода прямых
    double result_y = 0; // координата y точки схода прямых

    Point van_point_verticals; // точка схода прямых

    while (true)
    {
        capture >> src;

        // задаем roi, чтобы отсечь неинтересующие прямые
        int x_roi = src.cols / 4; // где начинается roi
        int width_roi = src.cols / 2;

        // получение углов градиента
        simpleSobel(src, grad_x, grad_y);

        // кластеризация по углам градиента
        Mat src_clustering(src.rows, src.cols, src.type(), Scalar(0, 0, 0));
        clustering(grad_x, grad_y, src_clustering);

        // векторизация границ кластеров
        vectorisation(src_clustering, src_vectorization);

        // отбираем вертикальные отрезки, преобразовав границы кластеров в отрезки
        Mat src_polylines(src.rows, src.cols, src.type(), Scalar(255, 255, 255));
        makePolylines(src_vectorization, src_polylines, 15, x_roi, width_roi);

        // выделяем контуры
        cvtColor(src_polylines, src_polylines, COLOR_BGR2GRAY);
        Canny(src_polylines, src_polylines, 30, 200);
        vector< vector<Point> > contours;
        findContours(src_polylines, contours, RETR_LIST, CHAIN_APPROX_NONE);

        // получение вектора коэффициентов { k,b } линейных функций x = k * y + b, полученных из линейной регрессии по контурам
        vector <LinearFunction> linearFunctions = linearRegressionForContours(contours);

        // получение вектора точек найденных вертикальных прямых
        vector < tuple<Point, Point> > vertical_lines = findVectorOfPointsOfVerticalLines(linearFunctions);

        // выделяем прямые по контурам методов Хафа
        //vector<Vec2f> lines = findLinesHough(src_polylines);
        // выбор вертикальных прямых
        //vector < tuple<Point, Point> > vertical_lines = selectionOfVerticalLines(lines);

        // оставляем прямые, которые вписываются с центральную часть ширины width_roi, которая начинается с x_roi
        roiForVerticalLines(vertical_lines, x_roi, width_roi);

        // отрисовка прямых
        drawLines(src, vertical_lines);

        // медианный фильтр
        if (medianCount % NUMBER_OF_MEDIAN_VALUES == 0)
        {
            // Если нужно провести медианный фильтр

            medianResult_x = medianFilter(valuesForMedianFilterX, NUMBER_OF_MEDIAN_VALUES);

            // так как медианный фильтр делаем по координатам x, надо координате x сопоставить соответствующий y
            for (int i = 0; i < NUMBER_OF_MEDIAN_VALUES - 1; i++)
            {
                if (valuesForMedianFilterX[i] == medianResult_x)
                {
                    van_point_verticals.x = medianResult_x;
                    van_point_verticals.y = valuesForMedianFilterY[i];
                    break;
                }
            }
        }
        else
        {
            //проверяем, что нашлось примерно поровну прямых слева и справа
            double quantityFactor = 1.5;
//            if (quantitativeFilter(vertical_lines, src.cols / 2, quantityFactor))
//            {
//                makeSpaceKB(result_x, result_y, vertical_lines);
//            }
            makeSpaceKB(result_x, result_y, vertical_lines);

            valuesForMedianFilterX[medianCount - 1] = result_x;
            valuesForMedianFilterY[medianCount - 1] = result_y;
        }

        // нахождение линий дорожной разметки
        vector< tuple<Point, Point> > roadMarkings = findRoadMarkingLines(src);

        // определение точки пересечения дорожной разметки
        Point van_point_lane = findVanishingPointLane(roadMarkings);

        // получение линии горизонта, размеченной вручную
        tuple<Point, Point> currentHorizonLine = manuallySelectingHorizonLine(src);
        Point currentHorizon_pt1 = get<0>(currentHorizonLine);
        Point currentHorizon_pt2 = get<1>(currentHorizonLine);

        tuple<Point, Point> accurateHorizonLine = getAccurateHorizonLine(currentHorizon_pt1.y, src.cols);
        Point accurateHorizon_pt1 = get<0>(accurateHorizonLine);
        Point accurateHorizon_pt2 = get<1>(accurateHorizonLine);

        tuple<Point, Point> accurateVerticalLine = getAccurateVerticalLine(van_point_lane.x, src.rows);
        Point accurateVertical_pt1 = get<0>(accurateVerticalLine);
        Point accurateVertical_pt2 = get<1>(accurateVerticalLine);

        line(src, van_point_lane, van_point_verticals, Scalar(255, 0, 0), 1, LINE_AA);
        //line(src, currentHorizon_pt1, currentHorizon_pt2, Scalar(0, 0, 255), 1, LINE_AA);

        // line(src, accurateVertical_pt1, accurateVertical_pt2, Scalar(255, 0, 0), 1, LINE_AA);
        // line(src, accurateHorizon_pt1, accurateHorizon_pt2, Scalar(0, 0, 255), 1, LINE_AA);

        double angle_vertical = angleBetweenStraightLines(van_point_lane, van_point_verticals, accurateVertical_pt1, accurateVertical_pt2);
        double angle_horizon = angleBetweenStraightLines(currentHorizon_pt1, currentHorizon_pt2, accurateHorizon_pt1, accurateHorizon_pt2);
        cout << abs(angle_vertical - angle_horizon) << endl;

        imshow("src", src);

        // увеличение счетчика для медианного фильтра
        if (medianCount % NUMBER_OF_MEDIAN_VALUES == 0)
        {
            medianCount = 1;
        }
        else
        {
            medianCount++;
        }

//        outputVideo << src;  // сохранение результата в файл

        // освобождаем память
        grad_x.release();
        grad_y.release();
        src_polylines.release();
        src_vectorization.release();
        src_clustering.release();
        src.release();

        int k = waitKey(25);
        if (k == 27)
        {
            // освобождаем память
            capture.release();
            break;
        }
    }
}

int main()
{
    const string PATH_test = "../videos/test.AVI";
    const string PATH_test2 = "../videos/test2.mp4";
    const string PATH_road = "../videos/road.mp4";
    const string PATH_road2 = "../videos/road2.mp4";
    const string PATH_road3 = "../videos/road3.mp4";

    //selectingLinesUsingHoughMethod(PATH_road3);
    selectingLinesUsingGradient(PATH_road3);
}