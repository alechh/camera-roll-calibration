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


/** Search for vertical straight lines on video using the Hough method
 *
 * @param src -- Input image.
 * @param resize -- image resizing factor.
 * @param delta -- Coefficient by which it is determined that the line is straight. The larger it is, the more lines will be selected.
 */
void findLinesHough(Mat &src, double resize = 1, int delta = 300)
{
    Mat dst;
    cvtColor(src,dst, COLOR_BGR2GRAY);
    Canny(src, dst, 50, 550, 3);

    vector<Vec2f> lines;
    set < tuple<double, double> > vertical_lines;

    HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0);

    cvtColor(dst, dst, COLOR_GRAY2BGR);

    set<tuple<double, double>> coefficientsKB;


    // draw lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho, theta, a, b, x0, y0;
        Point pt1, pt2;

        rho = lines[i][0];
        theta = lines[i][1];

        a = cos(theta);
        b = sin(theta);

        x0 = a * rho;
        y0 = b * rho;

        pt1.x = cvRound(x0 - 1000 * b);
        pt1.y = cvRound(y0 + 1000 * a);
        pt2.x = cvRound(x0 + 1000 * b);
        pt2.y = cvRound(y0 - 1000 * a);

        if (abs(pt1.x - pt2.x) < delta)
        {
            //--------------
            // Вставляю новый код: хочу прямой сопоставлять точку (k,b)
            // x = k * y + b2
            // Беру k со знаком -, потому что в OpenCV система координат устроена иначе
            double k = - double((pt2.x - pt1.x)) / (pt2.y - pt1.y);
            double b2 = pt1.x - pt1.y * double(pt2.x - pt1.x) / (pt2.y - pt1.y);

            // Почему-то работает так, хотя по идее должно быть наоборот
            //coefficientsKB.insert(make_tuple(k, b2));
            coefficientsKB.insert(make_tuple(b2, k));

            //--------------

            line(src, pt1, pt2, CV_RGB(255,0,0), 2, CV_AA);
            vertical_lines.insert(make_tuple(lines[i][0], lines[i][1]));
        }
    }
    //---------------------------------
    if (coefficientsKB.begin() != coefficientsKB.end())
    {
        double approaching_x = -1, approaching_y = -1;
        SpaceKB spaceKb(coefficientsKB);
        spaceKb.approaching_straight_line(approaching_x, approaching_y);
        if (approaching_x != -1 && approaching_y != -1)
        {
            Point pt;
            pt.x = approaching_x;
            pt.y = approaching_y;
            // circle(src, pt, 5, Scalar(255,255,0),5);
            cout<<"Вычисленная точка: ("<<approaching_x<<" ; "<<approaching_y<<" )"<<endl;
        }
        // spaceKb.print_points();
    }

    //---------------------------------

    //intersectionSearch(src, vertical_lines);
}

/** Function to test the function of finding straight vertical lines
 *
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
        findLinesHough(src);

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
    string PATH_test = "../videos/test.AVI";
    string PATH_test2 = "../videos/test2.mp4";
    string PATH_road = "../videos/road.mp4";
    string PATH_road2 = "../videos/road2.mp4";
    string PATH_road3 = "../videos/road3.mp4";

    simpleLineDetection(PATH_road3, 0.6);
    //simpleLineDetection(0, 0.6);
}