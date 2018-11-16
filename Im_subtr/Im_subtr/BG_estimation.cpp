// BG_estimation.cpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cstring>
#include "samples_utility.hpp"

#include<iostream>
#ifdef WINDOWS
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#endif

#include "Blob.h"
#include "BG_estimation.h"

using namespace std;
using namespace cv;

/*/ global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
*///////////////////////////////////////////////////////////////////////////////////////////////////

void BG(cv::Mat imgFrame1Copy, cv::Mat imgFrame2Copy, cv::Mat imgBackThresh, cv::Mat imgBack)
{	
	std::vector<Blob> blobs;

	cv::Mat imgCurr;
	cv::Mat imgDifference;
	cv::Mat imgThresh;
	cv::Mat imgThresh1;
	cv::Mat imgThresh11;
	cv::Mat imgBackThreshInv;
	cv::Mat imgBackNewPoints;

	imgCurr = imgFrame1Copy.clone();
	cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

	cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

	cv::Mat structuringElement3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat structuringElement5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Mat structuringElement7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::Mat structuringElement9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

	cv::erode(imgThresh, imgThresh, structuringElement3);
	cv::dilate(imgThresh, imgThresh, structuringElement9);
	cv::dilate(imgThresh, imgThresh, structuringElement9);
	cv::dilate(imgThresh, imgThresh, structuringElement9);

	cv::Mat imgThreshCopy = imgThresh.clone();

	std::vector<std::vector<cv::Point> > contours;

	cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

	cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);

	std::vector<std::vector<cv::Point> > convexHulls(contours.size());
	for (unsigned int i = 0; i < contours.size(); i++) {
		cv::convexHull(contours[i], convexHulls[i]);
	}

	for (auto &convexHull : convexHulls) {
		Blob possibleBlob(convexHull);
		blobs.push_back(possibleBlob);
	}

	cv::Mat imgConvexHulls(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

	convexHulls.clear();

	for (auto &blob : blobs) {
		convexHulls.push_back(blob.contour);
	}

	cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);

	cv::cvtColor(imgConvexHulls, imgThresh11, CV_BGR2GRAY);
	cv::bitwise_not(imgBackThresh, imgBackThreshInv);
	cv::bitwise_not(imgThresh11, imgThresh1);
	imgThresh11 = imgThresh1.mul(imgBackThreshInv / 255);
	imgBackNewPoints = imgCurr.mul(imgThresh11 / 255);
	cv::add(imgBack, imgBackNewPoints, imgBack);
	cv::threshold(imgBack, imgBackThresh, 1, 255.0, CV_THRESH_BINARY);
}
