#include <iostream>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "SpaceKB.h"

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
void makeSpaceKB(vector < tuple<Point, Point> > vertical_lines)
{
    set< tuple<double, double> > coefficientsKB;  // Коэффициенты b и k прямых x = ky+b для построения пространства Kb

    for (int i = 0; i < vertical_lines.size(); i++)
    {
        Point pt1, pt2;

        pt1 = get<0>(vertical_lines[i]);
        pt2 = get<1>(vertical_lines[i]);

        // x = k * y + b
        // Беру k со знаком -, потому что в OpenCV система координат инвертирована относительно оси OY
        double k = - double((pt2.x - pt1.x)) / (pt2.y - pt1.y);
        double b = pt1.x - pt1.y * double(pt2.x - pt1.x) / (pt2.y - pt1.y);

        coefficientsKB.insert(make_tuple(b, k));
        //coefficientsKB.insert(make_tuple(k, b));  // -- не работает
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
            cout << "Вычисленная точка: (" << approaching_x << " ; " << approaching_y << " )" << endl;
        }
        // spaceKb.print_points();
    }
}

/**
 * Select the vertival lines from the all lines
 * @param lines
 * @param delta
 * @return
 */
vector <tuple<Point, Point>> selectionOfVerticalLines(vector<Vec2f> lines, int delta = 300)
{
    vector <tuple<Point, Point>> vertical_lines;  // множество пар точек, через которые проходят вертикальные прямые

    for (int i = 0; i < lines.size(); i++)
    {
        double rho, theta;
        Point pt1, pt2;  // 2 точки, через которые проходит прямая

        rho = lines[i][0];
        theta = lines[i][1];

        calculatingPoints(rho, theta, pt1, pt2);  // вычисление двух точек прямой (pt1, pt2)

        if (abs(pt1.x - pt2.x) < delta)  // если прямая подозрительна на вертикальную
        {
            vertical_lines.emplace_back(pt1, pt2);
        }
    }
    return vertical_lines;
}

/**
 * Draw vertical lines
 * @param src -- Input image
 * @param lines -- vector of pairs of values rho and theta
 * @param delta -- coefficient by which it is determined whether a straight line is vertical
 */
void drawVerticalLines(Mat &src, vector <tuple<Point, Point>> vertical_lines)
{
    for (int i = 0; i < vertical_lines.size(); i++)
    {
        Point pt1, pt2;
        pt1 = get<0>(vertical_lines[i]);
        pt2 = get<1>(vertical_lines[i]);

        line(src, pt1, pt2, CV_RGB(255,0,0), 2, CV_AA);  // отрисовка прямой
    }
}

/**
 * Search for vertical straight lines on video using the Hough method
 * @param src -- Input image.
 * @param resize -- image resizing factor.
 * @param delta -- Coefficient by which it is determined that the line is straight.
 *                 The larger it is, the more lines will be selected.
 */
vector<Vec2f> findLinesHough(Mat &src, double resize = 1)
{
    Mat srcCopy;
    vector<Vec2f> lines;  // прямые, найденные на изображении

    cvtColor(src, srcCopy, COLOR_BGR2GRAY);  // Подготовка изображения для метода Хава поиска прямых
    Canny(srcCopy, srcCopy, 50, 550, 3);  // Подготовка изображения для метода Хава поиска прямых

    HoughLines(srcCopy, lines, 1, CV_PI/180, 150, 0, 0);

    return lines;
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
    while (true)
    {
        capture >> src;

        vector<Vec2f> lines = findLinesHough(src);  // нахождение прямых линий

        vector <tuple<Point, Point>> vertical_lines = selectionOfVerticalLines(lines);  // выбор только вертикальных линий

        drawVerticalLines(src, vertical_lines);  // отрисовка прямых линий

        makeSpaceKB(vertical_lines);  // построение пространства Kb, чтобы найти приближающую прямую через линейную регрессию

        if (resize != 1)
        {
            cv::resize(src, src, cv::Size(), resize, resize);
        }
        imshow("Result", src);

        int k = waitKey(25);
        if (k == 27)
        {
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

    simpleLineDetection(PATH_road3, 0.6);
}