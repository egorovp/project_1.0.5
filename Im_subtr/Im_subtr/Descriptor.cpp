// Descriptor.cpp

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
#include "Descriptor.h"

using namespace std;
using namespace cv;

cv::Mat structuringElement5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
Size subPixWinSize(10, 10), winSize(31, 31);
const int MAX_COUNT = 500;

namespace
{
	int descr_goodFeatures(cv::Mat cur_frame, cv::Mat prev_frame, cv::Mat bg, std::vector<description>& objects, bool& init)
	{
		vector<Point2f> points[4];
		vector<uchar> status;
		vector<float> err;

		cv::Mat imgDifference;
		cv::Mat imgThresh;

		std::vector<Blob> blobs;
		std::vector<std::vector<cv::Point> > possible_convexHull_prev;
		std::vector<std::vector<cv::Point> > possible_convexHull_new;
		std::vector<std::vector<cv::Point> > possible_convexHull_cur;

		// iteration for previous frame

		cv::absdiff(bg, prev_frame, imgDifference);

		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		cv::erode(imgThresh, imgThresh, structuringElement5);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::erode(imgThresh, imgThresh, structuringElement5);

		cv::Mat imgThreshCopy = imgThresh.clone();

		std::vector<std::vector<cv::Point>> contours;

		cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		std::vector<std::vector<cv::Point>> convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++)
		{
			cv::convexHull(contours[i], convexHulls[i]);
		}
		contours.clear();

		for (auto &convexHull1 : convexHulls)
		{
			Blob possibleBlob1(convexHull1);

			if (possibleBlob1.boundingRect.area() > 50 &&
				possibleBlob1.dblAspectRatio >= 0.2 &&
				possibleBlob1.dblAspectRatio <= 1.2 &&
				possibleBlob1.boundingRect.width > 15 &&
				possibleBlob1.boundingRect.height > 20 &&
				possibleBlob1.dblDiagonalSize > 30.0)
			{
				blobs.push_back(possibleBlob1);
				possible_convexHull_prev.push_back(convexHull1);
			}
		}

		// iteration for current frame

		cv::absdiff(bg, cur_frame, imgDifference);

		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		cv::erode(imgThresh, imgThresh, structuringElement5);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::erode(imgThresh, imgThresh, structuringElement5);

		imgThreshCopy = imgThresh.clone();

		std::vector<std::vector<cv::Point>> contours1;

		cv::findContours(imgThreshCopy, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		std::vector<std::vector<cv::Point>> convexHulls1(contours1.size());

		for (unsigned int i = 0; i < contours1.size(); i++)
		{
			cv::convexHull(contours1[i], convexHulls1[i]);
		}
		contours1.clear();
		for (auto &convexHull1 : convexHulls1)
		{
			Blob possibleBlob1(convexHull1);

			if (possibleBlob1.boundingRect.area() > 50 &&
				possibleBlob1.dblAspectRatio >= 0.2 &&
				possibleBlob1.dblAspectRatio <= 1.2 &&
				possibleBlob1.boundingRect.width > 15 &&
				possibleBlob1.boundingRect.height > 20 &&
				possibleBlob1.dblDiagonalSize > 30.0)
			{
				blobs.push_back(possibleBlob1);
				possible_convexHull_cur.push_back(convexHull1);
			}
		}

		// looking for new objects and recognition them

		cv::Mat imgFrame2lk = cur_frame.clone();
		cv::Mat imgFrame1lk = prev_frame.clone();
		//cv::cvtColor(imgFrame2lk, imgFrame2lk, CV_BGR2GRAY);
		//cv::cvtColor(imgFrame1lk, imgFrame1lk, CV_BGR2GRAY);

		if (init)// || objects.empty())
		{
			objects.clear();
			// automatic initialization
			goodFeaturesToTrack(imgFrame1lk, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			//cornerSubPix(imgFrame2lk, points[0], subPixWinSize, Size(-1, -1), termcrit);
			size_t i;
			size_t k = 0;
			std::vector<int> flag(possible_convexHull_prev.size());
			for (int j = 0; j < possible_convexHull_prev.size(); ++j)
			{
				for (i = 0; i < points[0].size(); ++i)
					if (cv::pointPolygonTest(possible_convexHull_prev[j], points[0][i], false) >= 0)
					{
						if (flag[j] == 0)
						{
							description new_obj;
							new_obj.index = j;
							new_obj.good_points.push_back(points[0][i]);
							new_obj.contour = possible_convexHull_prev[j];
							if (j % 3 == 0) new_obj.color = cv::Scalar(255.0 - (j / 3 * 10), 0.0, 0.0);
							if (j % 3 == 1) new_obj.color = cv::Scalar(0.0, 255.0 - (j / 3 * 10), 0.0);
							if (j % 3 == 2) new_obj.color = cv::Scalar(0.0, 0.0, 255.0 - (j / 3 * 10));
							objects.push_back(new_obj);
							points[0][k++] = points[0][i];
							flag[j]++;
						}
						else
						{
							if (objects.back().index == j)
							{
								objects.back().good_points.push_back(points[0][i]);
								points[0][k++] = points[0][i];
							}
							else
							{
								printf("\nIt may be error : (");
							}
						}
					}
			}
			points[0].resize(k);
			init = false;
			printf("\n\nobjects  -  %d\n\n", (int)objects.size());
		}
		else
		{
			printf("\nMiddle part");
			goodFeaturesToTrack(imgFrame1lk, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			int k = 0;
			std::vector<int> point_flag(points[0].size());
			for (int i = 0; i < points[0].size(); i++)
			{
				//if (i == points[0].size())
					//break;
				for (int j = 0; j < objects.size(); ++j)
					if (cv::pointPolygonTest(objects[j].contour, points[0][i], false) >= 0)
					{
						point_flag[i]++;
						objects[j].good_points.push_back(points[0][i]);
						//points[0].erase(points[0].begin() + i);
						//i--;
					}
				if(point_flag[i] == 0)
					points[0][k++] = points[0][i];
			}
			points[0].resize(k);

			printf("\n1st dangerous segment");

			for (auto &possible_new : possible_convexHull_prev)		//< \todo Возможно в будующем переделать на *_cur , чтобы новые объекты сразу выделялись
			{
				int flag = 0;
				for (int i = 0; i < points[0].size(); ++i)
				{
					if (cv::pointPolygonTest(possible_new, points[0][i], false) >= 0)
						flag++;
				}
				if (flag > 0)
					possible_convexHull_new.push_back(possible_new);
			}
			printf("\n2nd dangerous segment");
			if (!possible_convexHull_new.empty())
			{
				int old_size = objects.size();
				//cornerSubPix(imgFrame2lk, points[0], subPixWinSize, Size(-1, -1), termcrit);
				size_t i;






				std::vector<int> new_flag(possible_convexHull_new.size());
				int count_zeros = 0;
				for (int j = 0; j < possible_convexHull_new.size(); ++j)
				{
					int j1 = j + old_size;
					for (i = 0; i < points[0].size(); i++)
						if (cv::pointPolygonTest(possible_convexHull_new[j], points[0][i], false) >= 0)
						{
							if (new_flag[j] == 0)
							{
								description new_obj;
								new_obj.index = j1;
								new_obj.good_points.push_back(points[0][i]);
								new_obj.contour = possible_convexHull_new[j];
								if (j1 % 3 == 0) new_obj.color = cv::Scalar(255.0 - (j1 / 3 * 10), 0.0, 0.0);
								if (j1 % 3 == 1) new_obj.color = cv::Scalar(0.0, 255.0 - (j1 / 3 * 10), 0.0);
								if (j1 % 3 == 2) new_obj.color = cv::Scalar(0.0, 0.0, 255.0 - (j1 / 3 * 10));
								objects.push_back(new_obj);
								new_flag[j]++;
								//points[0].push_back(points[0][i]);
							}
							else
							{
								count_zeros = 0;
								for (int ind = 0; ind < j; ind++)
									if (new_flag[ind] == 0)
										count_zeros++;
								//printf("\ncount_zeros  -  %d", count_zeros);
								if (objects.size() > j1)
									objects[j1 - count_zeros].good_points.push_back(points[0][i]);
							}
						}
				}
			}
		}
		
		// vectors point[0] and objects filled

		if (!objects.empty())
		{
			printf("\nIt's last part.");
			//if (prev_frame.empty())
			//	imgFrame2lk.copyTo(prev_frame);
			for (int j = 0; j < objects.size(); ++j)
			{
				if (!objects[j].good_points.empty())
				{
					points[1].clear();
					status.clear();
					err.clear();
					calcOpticalFlowPyrLK(imgFrame1lk, imgFrame2lk, objects[j].good_points, points[1], status, err, winSize, 3, termcrit, 0, 0.001);
					size_t i;
					int fflag = 0;
					for (i = 0; i < points[1].size(); ++i)
					{

						if (!status[i])
							continue;
						for (int jj = 0; jj < possible_convexHull_cur.size(); ++jj)
							if (cv::pointPolygonTest(possible_convexHull_cur[jj], points[1][i], false) >= 0)		// \todo избавиться от выпуклых оболочек и перейти на контуры
							{
								objects[j].contour = possible_convexHull_cur[jj];
								fflag = 1;
							}
						if (fflag)
							break;
					}
				}
			}
			printf("\nEnd of last part.");
			//points[1].resize(k);
		}
		/*
		cv::absdiff(bg, cur_frame, imgDifference);

		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		cv::imshow("imgThresh", imgThresh);

		cv::erode(imgThresh, imgThresh, structuringElement5);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::erode(imgThresh, imgThresh, structuringElement5);

		cv::imshow("imgMorph", imgThresh);

		cv::Mat imgThreshCopy1 = imgThresh.clone();

		std::vector<std::vector<cv::Point>> contours;

		cv::findContours(imgThreshCopy1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat imgcontours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

		cv::drawContours(imgcontours, contours, -1, SCALAR_WHITE, -1);

		cv::imshow("imgContours", imgcontours);

		std::vector<std::vector<cv::Point>> convexHulls1(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++)
		{
			cv::convexHull(contours[i], convexHulls1[i]);
		}

		for (auto &convexHull1 : convexHulls1)
		{
			Blob possibleBlob1(convexHull1);

			if (possibleBlob1.boundingRect.area() > 50 &&
				possibleBlob1.dblAspectRatio >= 0.2 &&
				possibleBlob1.dblAspectRatio <= 1.2 &&
				possibleBlob1.boundingRect.width > 15 &&
				possibleBlob1.boundingRect.height > 20 &&
				possibleBlob1.dblDiagonalSize > 30.0)
			{
				blobs.push_back(possibleBlob1);
				possible_convexHull.push_back(convexHull1);
			}
		}

		cv::Mat imgFrame2lk = cur_frame.clone();
		cv::Mat imgFrame1lk = prev_frame.clone();
		cv::cvtColor(imgFrame2lk, imgFrame2lk, CV_BGR2GRAY);
		cv::cvtColor(imgFrame1lk, imgFrame1lk, CV_BGR2GRAY);
		if (true)
		{
			// automatic initialization
			goodFeaturesToTrack(imgFrame2lk, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			//cornerSubPix(imgFrame2lk, points[1], subPixWinSize, Size(-1, -1), termcrit);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				for (auto &possible_convexHull : possible_convexHull)
					if (cv::pointPolygonTest(possible_convexHull, points[1][i], false) >= 0)
						points[1][k++] = points[1][i];

			}
			points[1].resize(k);
		}
		else if (!points[0].empty())
		{
			if (prev_frame.empty())
				imgFrame2lk.copyTo(prev_frame);
			calcOpticalFlowPyrLK(imgFrame1lk, imgFrame2lk, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{

				if (!status[i])
					continue;

				points[1][k] = points[1][i];
				err[k++] = err[i];
			}
			points[1].resize(k);
			err.resize(k);
			/*
			vector<float> speed;
			float sp;
			int count;
			for (auto &possible_convexHull : possible_convexHull)
			{
				sp = 0;
				count = 0;
				for (i = k = 0; i < points[1].size(); i++)
					if (cv::pointPolygonTest(possible_convexHull, points[1][i], false) >= 0)
					{
						sp = sp + err[i];
						count++;
					}

				speed.push_back(sp / count);
				//circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			*/
		//}//*/
		return 0;
	}

	int descr_HOG(cv::Mat cur_frame, cv::Mat prev_frame, cv::Mat bg)
	{
		return 0;
	}

	int descr_pos(cv::Mat cur_frame, cv::Mat prev_frame, cv::Mat bg)
	{
		return 0;
	}
}

int recognition(cv::Mat cur_frame, cv::Mat prev_frame, cv::Mat bg, std::vector<description>& objects, bool& init, descriptor_type type)
{
	switch (type)
	{
	case POSITION_SPEED:
		descr_pos(cur_frame, prev_frame, bg);
		break;
	case HOG:
		return descr_HOG(cur_frame, prev_frame, bg);
		break;
	case ELSE:
		return descr_goodFeatures(cur_frame, prev_frame, bg, objects, init);
		break;
	default:
		return -1;
		break;
	}
}

int descriptor(std::vector<cv::Point> contour1, cv::Mat frame, descriptor_type type/*, std::vector<cv::Point> contour2, double eps*/)
{
	switch (type)
	{
	case POSITION_SPEED:
		break;
	case HOG:
		break;
	case ELSE:
		break;
	default:
		return -1;
		break;
	}
	/*
	std::vector<std::vector<cv::Point>> contours;
	contours.push_back(contour1);
	cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
	cv::drawContours(mask, contours, -1, cvScalar(1.0), -1);
	//cv::multiply(frame, mask, mask);
	cv::Scalar mean = cvScalar(0.0, 0.0, 0.0, 0.0);
	mean = cv::mean(mask);
	*/
	return 0;// mean[0];
}
