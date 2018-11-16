// ImageSubtractionCpp.sln
// main.cpp

//#include "opencv2/imgproc.hpp"

//#include <ctype.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cstring>
#include "samples_utility.hpp"
#include "opencv2/objdetect.hpp"


#include<iostream>
#ifdef WINDOWS
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#endif

#include "Blob.h"
#include "BG_estimation.h"
#include "Descriptor.h"

using namespace std;
using namespace cv;
/*
struct desription
{
	std::vector<cv::Point> good_points;
	cv::Point position;
	float speed;
	double mean;
};
*/
/*/ global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
*///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);
	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;
	bool start_track = false;
	vector<Point2f> points[4];
	cv::VideoCapture capVideo;
	vector<uchar> status;
	vector<float> err;
	cv::Mat imgFrame1;
	cv::Mat imgFrame2;
	cv::Mat backgroundMat[3];
	//capVideo.open("768x576.avi");
	cv::CommandLineParser parser(argc, argv, "{@input|0|}");
	string input = parser.get<string>("@input");
	if (input.size() == 1 && isdigit(input[0]))
		capVideo.open(input[0] - '0');
	else
		capVideo.open(input);
	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		std::cout << "\nerror reading video file" << std::endl << std::endl;      // show error message
#ifdef WINDOWS
		_getch();                    // it may be necessary to change or remove this line if not using Windows
#endif
		return(0);                                                              // and exit program
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		std::cout << "\nerror: video file must have at least two frames";
#ifdef WINDOWS
		_getch();
#endif
		return(0);
	}

	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);
	
	cv::Mat imgBack = cv::Mat::zeros(imgFrame1.rows, imgFrame1.cols, CV_8UC1);
	cv::Mat imgBackThresh = imgBack.clone();

	char chCheckForEscKey = 0;
	int countFrames = 2;
	std::vector<description> objects;
	//std::vector<std::vector<cv::Point> > new_objects;
	//std::vector<std::vector<cv::Point> > old_objects;
	std::vector<int> number;
	bool init = true;
	
	// \todo delete this flag
	int flag = 0;
	
	while (capVideo.isOpened() && chCheckForEscKey != 27) {

		std::vector<Blob> blobs;
		std::vector<Blob> blobs1;
		std::vector<int> obj_numer;

		cv::Mat imgFrame1Copy = imgFrame1.clone();
		cv::Mat imgFrame2Copy = imgFrame2.clone();
		cv::Mat imgThresh;
		cv::Mat imgDifference;
		cv::Mat imgFrame22Copy;

		cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
		cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
		cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

		cv::Mat structuringElement3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::Mat structuringElement5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::Mat structuringElement7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::Mat structuringElement9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

		//std::vector<std::vector<cv::Point> > possible_convexHull;

		if (countFrames < 20)
		{
			BG(imgFrame1Copy, imgFrame2Copy, imgBackThresh, imgBack);
		}
		cv::imshow("BackGround", imgBack);
		/*
		//cv::imshow("BG", imgBack);
		
		//if (countFrames == 40)
		//{
		//	imwrite("background.png", imgBack);
		//}
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cv::absdiff(imgBack, imgFrame2Copy, imgDifference);

		cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		//cv::imshow("imgThresh", imgThresh);


		cv::erode(imgThresh, imgThresh, structuringElement5);
		//cv::dilate(imgThresh, imgThresh, structuringElement3);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::dilate(imgThresh, imgThresh, structuringElement5);
		cv::erode(imgThresh, imgThresh, structuringElement5);


		//cv::imshow("imgThresh", imgThresh);
		//cv::imshow("imgMorph", imgThresh);

		cv::Mat imgThreshCopy1 = imgThresh.clone();

		std::vector<std::vector<cv::Point> > contours1;

		cv::findContours(imgThreshCopy1, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat imgContours1(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

		cv::drawContours(imgContours1, contours1, -1, SCALAR_WHITE, -1);

		//cv::imshow("imgContours", imgContours1);

		std::vector<std::vector<cv::Point> > convexHulls1(contours1.size());

		for (unsigned int i = 0; i < contours1.size(); i++) {
			cv::convexHull(contours1[i], convexHulls1[i]);
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
				blobs1.push_back(possibleBlob1);
				possible_convexHull.push_back(convexHull1);
			}
		}

		for (auto &contour : contours1) {
			Blob possibleBlob1(contour);

			if (possibleBlob1.boundingRect.area() > 50 &&
				possibleBlob1.dblAspectRatio >= 0.2 &&
				possibleBlob1.dblAspectRatio <= 1.2 &&
				possibleBlob1.boundingRect.width > 15 &&
				possibleBlob1.boundingRect.height > 20 &&
				possibleBlob1.dblDiagonalSize > 30.0)
			{
				new_objects.push_back(contour);
			}
		}*/
		
		if (countFrames >= 20)
			recognition(imgFrame2Copy, imgFrame1Copy, imgBack, objects, init, ELSE);

		printf("\nobjects  -  %d\n init  -  %d\n", objects.size(), init);

		imgFrame22Copy = imgFrame2.clone();
		if (!objects.empty())
			for (int i = 0; i < objects.size(); ++i)
			{
				if (!objects[i].contour.empty())
				{
					{
						cv::rectangle(imgFrame22Copy, cv::boundingRect(objects[i].contour), objects[i].color, 2);             // draw a red box around the blob
						//cv::circle(imgFrame2Copy, blob1.centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center
					}
				}
			}
		/*
		//cv::HOGDescriptor()

		// здесь тест goodFeaturesToTrack
		//if (countFrames == 20)
		//	needToInit = true;
		//else
		//	needToInit = false;
				
		cv::Mat imgFrame2lk = imgFrame2.clone();
		cv::Mat imgFrame1lk = imgFrame1.clone();
		cv::cvtColor(imgFrame2lk, imgFrame2lk, CV_BGR2GRAY);
		cv::cvtColor(imgFrame1lk, imgFrame1lk, CV_BGR2GRAY);
		if (needToInit)
		{
			// automatic initialization
			goodFeaturesToTrack(imgFrame2lk, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
			//cornerSubPix(imgFrame2lk, points[1], subPixWinSize, Size(-1, -1), termcrit);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				for (auto &possible_convexHull : possible_convexHull)
					if (cv::pointPolygonTest(possible_convexHull, points[1][i], false)>=0)
						points[1][k++] = points[1][i];

			}
			points[1].resize(k);
		}
		else if (!points[0].empty())
		{
			if (imgFrame1.empty())
				imgFrame2lk.copyTo(imgFrame1);
			calcOpticalFlowPyrLK(imgFrame1lk, imgFrame2lk, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
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
				
				speed.push_back(sp/count);
				//circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
		}



		cv::Mat imgConvexHulls1(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

		convexHulls1.clear();

		for (auto &blob1 : blobs1) {
			convexHulls1.push_back(blob1.contour);
		}

		cv::drawContours(imgConvexHulls1, convexHulls1, -1, SCALAR_WHITE, -1);

		// test drawing one contour
		cv::Mat one_contour = cv::Mat::zeros(imgBack.size(), CV_8UC1);
		cv::drawContours(one_contour, new_objects, 0, SCALAR_WHITE, -1);
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//cv::imshow("imgConvexHulls", //imgConvexHulls1// one_contour);


		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

		for (auto &blob1 : blobs1) {                                                  // for each blob
			cv::rectangle(imgFrame2Copy, blob1.boundingRect, SCALAR_RED, 2);             // draw a red box around the blob
			cv::circle(imgFrame2Copy, blob1.centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center
		}
		int i_max = old_objects.size();
		int j_max = new_objects.size();
		std::vector<int> del;
		for (int j = 0; j < j_max; ++j)
			for (int i = 0; i < i_max; ++i)
			{
				//
				if (abs(descriptor(old_objects[i], imgFrame2Copy, POSITION_SPEED) - descriptor(new_objects[j], imgFrame1Copy, POSITION_SPEED)) < 10)
				{
					//old_objects[i] = new_objects[j];
					//number[i] = j;
					//new_objects.erase(new_objects.begin() + j);
					//del.push_back(j);
					//j--;
					//j_max--;
				}
				//
			}
		//
		for (int i = 0; i < del.size(); ++i)
		{
			new_objects.erase(new_objects.begin() + del.back());
			del.pop_back();
		}
		//
		//if (flag < 12)
		{
			// если объекты не распознаны, то они новые
			int obj_number = old_objects.size();
			for (int i = 0; i < new_objects.size(); ++i)
			{
				old_objects.push_back(new_objects[i]);
				number.push_back(obj_number + i);
			}
			//flag++;
		}






		for (int i = 0; i < old_objects.size(); ++i)
		{
			char buffer[25];
			cv::Rect boundingRectangle = cv::boundingRect(old_objects[i]);
			sprintf_s(buffer, "%d", number[i]);
			cv::putText(imgFrame2Copy, buffer, cv::Point(boundingRectangle.x - 5, boundingRectangle.y - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, CV_AA);
		}
		//
		for (int i = 0; i < blobs1.size(); ++i)
		{
			char buffer[25];
			//cv::Rect boundingRectangle = cv::boundingRect(new_objects[i]);
			sprintf_s(buffer, "%d", i);
			cv::putText(imgFrame2Copy, buffer, cv::Point(blobs1[i].boundingRect.x - 5, blobs1[i].boundingRect.y - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1, CV_AA);
		}
		//


		if (!points[1].empty())
		{
			//
			if (countFrames > 40)
			{
				char buffer[25];
				//CvFont font;
				//cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5);
				int speed = 234567;
				//float speed = (points[1][0].x - points[2][0].x) * (points[1][0].x - points[2][0].x) + (points[1][0].y - points[2][0].y) * (points[1][0].y - points[2][0].y);
				//printf("\n speed = %lf\n", speed);
				float speed2 = 100;
				if (status[0])
					speed2 = err[0];
				sprintf_s(buffer, "speed = %f", speed2);
				putText(imgFrame2Copy, buffer, points[1][0], FONT_HERSHEY_DUPLEX, 1, Scalar(255, 0, 0), 1);
			}
			//
			for (int i = 0; i < points[1].size(); i++)
			{
				//points[1][k++] = points[1][i];
				circle(imgFrame2Copy, points[1][i], 3, Scalar(70, 200, 200), -1, 8);
			}
		}
		*/
		char buffer[25];
		//cv::Rect boundingRectangle = cv::boundingRect(old_objects[i]);
		sprintf_s(buffer, "%d", countFrames);
		cv::putText(imgFrame22Copy, buffer, cv::Point(15, 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1, CV_AA);
		cv::imshow("imgFrame22Copy", imgFrame22Copy);

		// now we prepare for the next iteration
		//std::swap(points[1], points[0]);
		imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is
		
		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT))       // if there is at least one more frame
		{
			capVideo.read(imgFrame2);
			// read it
			countFrames++;
		}
		else
		{                                                  // else
			std::cout << "end of video\n";                      // show end of video message
			break;                                              // and jump out of while loop
		}
		needToInit = false;
		char next_frame = 0;
		/*
		while (true)
		{
			next_frame = (char)waitKey(0);
			if (next_frame != 0)
				break;
		}
		*/
		char c = (char)waitKey(1);
		//c = next_frame;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		}
		chCheckForEscKey = c;      // get key press in case user pressed esc

	}

	if (chCheckForEscKey != 27) 
	{               // if the user did not press esc (i.e. we reached the end of the video)
		cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}

	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

	return(0);
}
