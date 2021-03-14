#include <iostream>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "SpaceKB.h"

using namespace cv;
using namespace std;


/** Simple optical flow: Rarneback method
 *
 * @tparam T
 * @param path -- path to the video file. 0 means that the video will be read from the webcam.
 * @param resize -- image resizing factor.
 */
template <class T>
void simpleRarnebackMethod(T path, double resize = 1)
{
    VideoCapture capture(path);
    if (!capture.isOpened())
    {
        cerr << "Unable to open file!" << endl;
        return;
    }
    Mat frame1, prvs;
    capture >> frame1;
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);

/*    // Save result video
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("outRarneback.avi",VideoWriter::fourcc('M','J','P','G'),10, Size(frame_width,frame_height),true);*/

    while (true)
    {
        Mat frame2, next;
        capture >> frame2;
        if (frame2.empty())
        {
            break;
        }
        cvtColor(frame2, next, COLOR_BGR2GRAY);
        Mat flow(prvs.size(), CV_32FC2);
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // visualization
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        //build hsv image
        Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);

        if (resize != 1)
        {
            cv::resize(bgr, bgr, cv::Size(), resize, resize);
        }
        imshow("frame2", bgr);

/*        // Save video
        video.write(bgr);*/

        int keyboard = waitKey(30);
        if (keyboard == 27)
        {
            break;
        }
        prvs = next;
    }
}


/** Simple optical flow: Lucas-Canade method
 *
 * @tparam T
 * @param path -- path to the video file. 0 means that the video will be read from the webcam.
 * @param resize -- image resizing factor.
 */
template <class T>
void simpleLucasCanadeMethod(T path, double resize = 1)
{
    VideoCapture capture(path);

    if (!capture.isOpened())
    {
        cerr << "Unable to open file!" << endl;
        return;
    }

    std::vector<Scalar> colors;
    RNG rng;
    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }

    Mat old_frame, old_gray;
    std::vector<Point2f> p0, p1;

    capture >> old_frame;
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

/*    // Save result video
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("outLucasCanade.avi",VideoWriter::fourcc('M','J','P','G'),10, Size(frame_width,frame_height),true);*/

    while (true)
    {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
        {
            break;
        }
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        std::vector<uchar> status;
        std::vector<float> err;
        //TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(21,21), 5);

        std::vector<Point2f> good_new;
        for (uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 1)
            {
                good_new.push_back(p1[i]);
                // draw the tracks
                line(mask,p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }
        Mat img;
        add(frame, mask, img);

        if (resize != 1)
        {
            cv::resize(img, img, cv::Size(), resize, resize);
        }
        imshow("Frame", img);

/*        // Save video
        video.write(img);*/

        old_gray = frame_gray.clone();
        p0 = good_new;

        int k = waitKey(25);
        if (k == 27)
        {
            return;
        }

    }
}


/** Function for sorting contours from biggest to smallest, used in the Contours_Detection method
 *
 * @param contour1.
 * @param contour2.
 * @return true, if the area of the contour1 bigger than the area of the contour2. Else return false.
 */
bool compareContourAreas(vector<Point> contour1, vector<Point> contour2)
{
    cv::Rect rect1 = cv::boundingRect(contour1);
    cv::Rect rect2 = cv::boundingRect(contour2);
    return (rect1.area() > rect2.area());
}

/** Tracking and edge contours on video
 *
 * @tparam T
 * @param path -- path to the video file. 0 means that the video will be read from the webcam.
 * @param resize -- image resizing factor.
 */
template <class T>
void contoursDetection(T path, double resize = 1)
{
    VideoCapture capture(path);
    if (!capture.isOpened())
    {
        cerr<<"Error"<<endl;
        return;
    }

    Mat img;
    while (true)
    {
        capture >> img;
        //Mat drawing = img.clone();
        Mat drawing = Mat::zeros(img.size(), CV_8UC3); // If you wanna see contours on the source video, comment this line and uncomment previous line
        Mat source = img.clone();

        if (resize != 1)
        {
            cv::resize(source, source, cv::Size(), resize, resize);
        }
        imshow("Source", source);

        cv::GaussianBlur(img, img, Size(3,3), 3);
        Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(1, 1));
        morphologyEx(img, img, MORPH_GRADIENT, element);

        cvtColor(img, img, COLOR_BGR2GRAY);

        vector< vector<Point> > contours;
        vector<Vec4i> hierarchy;

        Canny(img, img, 20, 100, 3);
        findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

        //sort(contours.begin(), contours.end(), compareContourAreas); // sort contours from biggest to smallest

        for (int i = 0; i < contours.size(); i++)
        {
            Rect brect = boundingRect(contours[i]);
            if (brect.area() < 1000)
            {
                continue;
            }

            drawContours(drawing, contours, i, CV_RGB(255,0,0), 1, 8, hierarchy, 0, Point());
        }

        if (resize != 1)
        {
            cv::resize(drawing, drawing, cv::Size(), resize, resize);
        }
        imshow("Contours", drawing);

        int k = waitKey(25);
        if (k == 27)
        {
            break;
        }
    }
}


/** Finding the intersection of lines
 *
 * @param src -- Input image
 * @param lines -- set of lines (<rho, theta>)
 */
void intersectionSearch(Mat &src, set< tuple<double, double> > lines)
{
    float rho1, theta1, rho2, theta2;
    Point pt;
    set< tuple<double, double> >::iterator i,j;
    for (i = lines.begin(); i != lines.end(); i++)
    {
        for (j = i; j != lines.end(); j++)
        {
            rho1 = get<0>(*i);
            rho2 = get<0>(*j);
            theta1 = get<1>(*i);
            theta2 = get<1>(*j);
            pt.x = (sin(theta2) * rho1 - sin(theta1) * rho2) / (cos(theta1) * sin(theta2) - sin(theta1) * cos(theta2));
            pt.y = (-cos(theta2) * rho1 + cos(theta1) * rho2) / (cos(theta1) * sin(theta2) - sin(theta1) * cos(theta2));
            if (0 < pt.x && pt.x < src.cols && 0 < pt.y && pt.y < src.rows)
            {
                circle(src,pt, 5, Scalar(0,255,0),5);
            }
//            else
//            {
//                cout<<"ПЕРЕСЕЧЕНИЕ ("<<pt.x<<";"<<pt.y<<")"<<endl;
//            }
        }
    }
}


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

            //-----------

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
            circle(src, pt, 5, Scalar(255,255,0),5);
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
        imshow("Lines", src);

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

    //simpleRarnebackMethod(PATH_road3, 0.4);
    //simpleLucasCanadeMethod(PATH_road2, 0.4);
    //contoursDetection(PATH_road3, 0.4);
    simpleLineDetection(PATH_road3, 0.6);
}