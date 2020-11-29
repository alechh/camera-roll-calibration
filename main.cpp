#include <iostream>
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;


/** Optical flow: Rarneback method

@param path -- path to the video file. 0 means that the video will be read from the webcam.
@param resize -- image resizing factor.
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


int main()
{
    string PATH_test = "../videos/test.AVI";
    string PATH_test2 = "../videos/test2.mp4";
    string PATH_road = "../videos/road.mp4";
    string PATH_road2 = "../videos/road2.mp4";
    string PATH_road3 = "../videos/road3.mp4";

    Rarneback(PATH_road, 0.4);
}