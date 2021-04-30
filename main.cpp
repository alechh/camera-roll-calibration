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
void makeSpaceKB(double &result_x, vector <tuple<Point, Point>> vertical_lines)
{
    set< tuple<double, double> > coefficientsKB;  // Коэффициенты b и k прямых x = ky+b для построения пространства Kb

    for (auto & vertical_line : vertical_lines)
    {
        Point pt1, pt2;

        pt1 = get<0>(vertical_line);
        pt2 = get<1>(vertical_line);

        // x = k * y + b
        // Беру k со знаком -, потому что в OpenCV система координат инвертирована относительно оси OY
        double k = - double((pt2.x - pt1.x)) / (pt2.y - pt1.y);
        double b = pt1.x - pt1.y * double(pt2.x - pt1.x) / (pt2.y - pt1.y);

        coefficientsKB.insert(make_tuple(b, k));
    }

    if (coefficientsKB.begin() != coefficientsKB.end())  // если нашлось хотя бы 2 прямые
    {
        double approaching_x = -1;
        double approaching_y = -1;

        SpaceKB spaceKb(coefficientsKB);

        spaceKb.approaching_straight_line(approaching_x, approaching_y);  // вычисление координат точки пересечения прямых

        if (approaching_x != -1 && approaching_y != -1)
        {
            Point pt;
            pt.x = approaching_x;
            pt.y = approaching_y;

            result_x = approaching_x;

            // cout << "Вычисленная точка: (" << approaching_x << " ; " << approaching_y << " )" << endl;
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

        cv::line(src, pt1, pt2, CV_RGB(255,0,0), 1, CV_AA);  // отрисовка прямой
    }
}

/**
 * Search for vertical straight lines on video using the Hough method
 * @param src -- Input image.
 * @param resize -- image resizing factor.
 * @param delta -- Coefficient by which it is determined that the line is straight.
 *                 The larger it is, the more lines will be selected.
 */
vector<Vec2f> findLinesHough(Mat &src)
{
    Mat src_gray, src_canny, src_8UC1;
    vector<Vec2f> lines;  // прямые, найденные на изображении

    // Media blur-----------------------
//    imshow("after_blurred", src);
//     int n = 5;
//     medianBlur(src, src, n);
//    imshow("blurred", src);
    //----------------------------------

    //cvtColor(src, src_gray, COLOR_BGR2GRAY);  // Подготовка изображения для метода Хафа поиска прямых
    Canny(src, src_canny, 50, 200, 3);  // Подготовка изображения для метода Хафа поиска прямых

    //imshow("canny", src_canny);

    //HoughLines(srcCopy, lines, 1, CV_PI/180, 150, 0, 0);
    HoughLines(src_canny, lines, 1, CV_PI / 180, 50, 0, 0);

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
    //imshow("src", src);
}

/**
 * Put x coordinate of the point on the image
 * @param src -- Input image
 * @param x -- x coordinate
 */
void showXOnImage(Mat &src, double x)
{
    String text = to_string(x);
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 2;
    int thickness = 3;
    int baseline=0;
    Size textSize = getTextSize(text, fontFace,
                                fontScale, thickness, &baseline);
    baseline += thickness;

    Point textOrg(15,50);

    putText(src, text, textOrg, fontFace, fontScale,
            Scalar(0, 0, 0), thickness, 8);
}

template <class T>
void bubbleSort(T *values, int size)
{
    for (size_t i = 0; i + 1 < size; i++)
    {
        for (size_t j = 0; j + 1 < size - i; j++)
        {
            if (values[j + 1] < values[j])
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
 * Function of finding straight vertical lines
 * @tparam T
 * @param path -- path to the video file. 0 means that the video will be read from the webcam.
 * @param resize -- image resizing factor.
 */
template <class T>
void simpleLineDetection(T path, double resize = 1)
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

        makeSpaceKB(result_x, vertical_lines);  // построение пространства Kb, чтобы найти приближающую прямую через
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

void makePolylines(Mat &src, Mat &dst, int delta = 10)
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
                if (newColor == Vec3b(0, 0, 0))
                {
                    // встретили черный цвет, значит это начало нового отрезка
                    // Есть идея: склеивать близкие отрезки (по delta)
                    if (abs(j - end) < delta)
                    {
                        // end -- конец предыдущего найденного отрезка
                        // если есть 2 отрезка в одной колонке, которые очень близко, но разрвны с разрывом длиной delta
                        // тогда мы их соединяем
                        begin = end + 1;
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
    while(currPolyline)
    {
        // TODO надо подумать над длиной (дальние отрезки пока не выводятся, а хотелось бы)
        if (30 <= currPolyline->length() && currPolyline->length() <= 500)
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

void vectorisation(Mat &src, Mat &dst)
{
    auto *listOfIntervalsLists = new ListOfIntervalsLists;
    int begin = 0;
    int end;
    int cluster_num = 1;

    // заполняем список списков интервалов для первой строки
    for (int j = 1; j < src.cols; j++)
    {
        if (src.at<Vec3b>(0, j) != src.at<Vec3b>(0, j - 1))
        {
            end = j - 1;
            auto *intervalList = new IntervalsList;
            intervalList->addInterval(begin, end, 0, cluster_num, src.at<Vec3b>(0, j - 1));
            begin = j;
            cluster_num++;

            listOfIntervalsLists->addIntervalList(intervalList);
        }
    }
    // последний интервал в первой строке
    auto *intervalList = new IntervalsList;
    intervalList->addInterval(begin, src.cols - 1, 0, cluster_num, src.at<Vec3b>(0, src.cols - 1));
    cluster_num++;
    listOfIntervalsLists->addIntervalList(intervalList);

    // заполняем списки интервалов для остальных строк
    for (int i = 1; i < src.rows; i++)
    {
        // заполним первый интервал мусорными значениями (не добавляем его в список)
        auto *currInterval = new Interval(-1, -1, -1, -1, src.at<Vec3b>(0, 0));

        IntervalsList *currIntervalList = listOfIntervalsLists->head;

        int prevIntervalEnd = currIntervalList->tail->end;  // эта переменная нужна для своевременного сдвига текущего списка интервалов

        begin = 0;
        cluster_num = 1;

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
                currInterval = new Interval(begin, end, i, cluster_num, color);
                begin = j;
                cluster_num++;

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
        cluster_num++;
        currInterval = new Interval(begin, src.cols - 1, i, cluster_num, src.at<Vec3b>(i, src.cols - 1));
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

    while (true)
    {
        capture >> src;

        // получение углов градиента
        simpleSobel(src, grad_x, grad_y);

        // кластеризация по углам градиента
        Mat src_clustering(src.rows, src.cols, src.type());
        clustering(grad_x, grad_y, src_clustering);

        // векторизация границ кластеров
        vectorisation(src_clustering, src_vectorization);

        // отбираем вертикальные отрезки
        Mat src_polylines(src.rows, src.cols, src.type(), Scalar(255, 255, 255));
        makePolylines(src_vectorization, src_polylines, 0);

        // выделяем контуры
        cvtColor( src_polylines, src_polylines, COLOR_BGR2GRAY );
        Canny(src_polylines, src_polylines, 50, 200); // TODO подумать над коэффициентами
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours( src_polylines, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE );

        // рисуем контуры
        Mat src_contour(src.rows, src.cols, src.type(), Scalar(255, 255, 255));
        for (int i = 0; i < contours.size(); i++)
        {
            drawContours(src_contour,contours, i, Scalar(0, 0, 0), -1);
        }

        // выделяем прямые по контурам методов Хафа
        vector<Vec2f> lines = findLinesHough(src_contour);  // нахождение прямых линий
        vector < tuple<Point, Point> > vertical_lines = selectionOfVerticalLines(lines);  // выбор вертикальных прямых
        drawLines(src, vertical_lines);  // отрисовка прямых

        imshow("src", src);

        // задаём ROI
//        int x = src.cols / 3;
//        int y = 0;
//        int width = src.cols / 3;
//        int height = src.rows;
//        Mat dst_roi = dst(Rect(x, y, width, height));

        // сдвиг координат из-за roi
//        for (auto & line : vertical_lines)
//        {
//            get<0>(line).x += width;
//            get<1>(line).x += width;
//        }

        //outputVideo << src;  // сохранение результата в файл

        // освобождаем память
        grad_x.release();
        grad_y.release();
        src_polylines.release();
        src_vectorization.release();
        src_clustering.release();
        src_contour.release();
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

    //simpleLineDetection(PATH_road3);
    selectingLinesUsingGradient(PATH_road3);
}