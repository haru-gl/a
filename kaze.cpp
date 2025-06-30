#include <opencv2/opencv.hpp>
#include "kaze.h"

void kaze::set_parameters(cv::Ptr<cv::KAZE>& detector)
{
	detector->setExtended(false);					//false
	detector->setUpright(false);					//false
	detector->setThreshold(kaze_threshold);			//0.001f
	detector->setNOctaves(kaze_nOctaves);			//4
	detector->setNOctaveLayers(kaze_nOctaveLayers);	//4
	detector->setDiffusivity(cv::KAZE::DIFF_PM_G2);
}

void kaze::set_kazeTh(double th)
{
	kaze_threshold = th;
}

double kaze::get_kazeTh(void) const
{
	return kaze_threshold;
}

void kaze::featuredetection(clipedmap_data& cm, target_data& td)
{
	cv::Ptr<cv::KAZE> detector = cv::KAZE::create();
	set_parameters(detector);
	detector->detectAndCompute(cm.oImage, cv::Mat(), cm.oPts, cm.oFeatures);
	detector->detectAndCompute(td.oImage, cv::Mat(), td.oPts, td.oFeatures);
//	kazeDetector->detect(src, keypnts);
//	cv::Mat dstKaze;
//	cv::drawKeypoints(src, keypnts, dstKaze, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	cv::imshow("KAZE", dstKaze);
}
