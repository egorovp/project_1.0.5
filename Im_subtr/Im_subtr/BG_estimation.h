//#pragma once
// BG_estimation.h

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

/* BackGround estimation
 * in		prev_frame
 * in		cur_frame
 * in/out	bg_mask
 * in/out	bg
 */
void BG(cv::Mat prev_frame, cv::Mat cur_frame, cv::Mat fulling_mask, cv::Mat bg);

// можно добавить маску, в которой будет учитываться записывался пиксель или нет (1/0)
