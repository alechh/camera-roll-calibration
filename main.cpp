#include <iostream>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc/imgproc_c.h>

using namespace cv;
using namespace std;


/** Simple optical flow: Rarneback method
 *
 * @tparam T
 * @param path -- path to the video file. 0 means that the video will be read from the webcam.
 * @param resize -- image resizing factor.
 */
template <class T>
void Rarneback(T path, double resize = 1)
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
void Lucas_Canade(T path, double resize = 1)
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
void Contours_Detection(T path, double resize = 1)
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


/** Search for vertical straight lines on video using the Hough method
 *
 * @param src -- Input image.
 * @param resize -- image resizing factor.
 * @param delta -- Coefficient by which it is determined that the line is straight. The larger it is, the more lines will be selected.
 */
void find_lines_Hough(Mat &src, double resize = 1, int delta = 300)
{
    Mat dst;
    cvtColor(src,dst, COLOR_BGR2GRAY);
    Canny(src, dst, 50, 550, 3);

    vector<Vec2f> lines;

    HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0);

    cvtColor(dst, dst, COLOR_GRAY2BGR);

    // draw lines
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho,theta, a, b, x0, y0;
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
            line(src, pt1, pt2, CV_RGB(255,0,0), 2, CV_AA);
        }
    }
}


/** Function to test the function of finding straight vertical lines
 *
 * @tparam T
 * @param path -- path to the video file. 0 means that the video will be read from the webcam.
 * @param resize -- image resizing factor.
 */
template <class T>
void simple_line_detection(T path, double resize = 1)
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
        find_lines_Hough(src);

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

    //Rarneback(PATH_road, 0.4);
    //Lucas_Canade(PATH_road2, 0.4);
    //Contours_Detection(PATH_road2, 0.4);
    simple_line_detection(PATH_road2, 0.4);
}