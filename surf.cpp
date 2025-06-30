#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "surf.h"

void surf::set_surfTh(double th)
{
	surf_Threshold = th;
}

double surf::get_surfTh(void) const
{
	return surf_Threshold;
}

void surf::featuredetection(clipedmap_data& cm, target_data& td)
{
	cv::Ptr<cv::xfeatures2d::SURF> surfDetector = cv::xfeatures2d::SURF::create();
	surfDetector->setHessianThreshold(surf_Threshold);
	surfDetector->detectAndCompute(cm.oImage, cv::Mat(), cm.oPts, cm.oFeatures);
	surfDetector->detectAndCompute(td.oImage, cv::Mat(), td.oPts, td.oFeatures);
}

