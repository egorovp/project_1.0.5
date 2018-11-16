// Descriptor.h

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

enum descriptor_type
{
	POSITION_SPEED,
	HOG,
	ELSE
};

struct description
{
	int index;
	cv::Scalar color;
	std::vector<cv::Point2f> good_points;
	std::vector<cv::Point> contour;
	cv::Point position;
	float speed;
	double mean;
};

/* BackGround estimation
* in		prev_frame
* in		cur_frame
* in/out	bg_mask
* in/out	bg
*/
int descriptor(std::vector<cv::Point> contour1, cv::Mat frame, descriptor_type type/*, std::vector<cv::Point> contour2, double eps*/);

int recognition(cv::Mat cur_frame, cv::Mat prev_frame, cv::Mat bg, std::vector<description>& objects, bool& init, descriptor_type type);
