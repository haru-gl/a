#include <opencv2/opencv.hpp>
#include "brisk.h"

void brisk::set_brisk_Th(int th)
{
	brisk_Threshold = th;
}

int brisk::get_brisk_Th(void) const
{
	return brisk_Threshold;
}

void brisk::set_brisk_parameters(cv::Ptr<cv::BRISK>& detector)
{
	detector->setOctaves(brisk_Octaves);
	detector->setThreshold(brisk_Threshold);
}

void brisk::featuredetection(clipedmap_data& cm, target_data& td)
{
	cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
	set_brisk_parameters(detector);
	detector->detectAndCompute(cm.oImage, cv::Mat(), cm.oPts, cm.oFeatures);
	detector->detectAndCompute(td.oImage, cv::Mat(), td.oPts, td.oFeatures);
}
