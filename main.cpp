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


/** Simple optical flow: Rarneback method

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


/** Simple optical flow: Lucas-Canade method

@param path -- path to the video file. 0 means that the video will be read from the webcam.
@param resize -- image resizing factor.
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


int main()
{
    string PATH_test = "../videos/test.AVI";
    string PATH_test2 = "../videos/test2.mp4";
    string PATH_road = "../videos/road.mp4";
    string PATH_road2 = "../videos/road2.mp4";
    string PATH_road3 = "../videos/road3.mp4";

    //Rarneback(PATH_road, 0.4);
    Lucas_Canade(PATH_road2, 0.4);
}