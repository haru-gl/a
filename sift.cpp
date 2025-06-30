#include <opencv2/opencv.hpp>
#include "sift.h"

void sift::set_sift_contrastTh(double th)
{
	contrastThreshold = th;
}

double sift::get_sift_contrastTh(void) const
{
	return contrastThreshold;
}

void sift::set_sift_egdeTh(double th)
{
	edgeThreshold = th;
}

double sift::get_sift_edgeTh(void) const
{
	return edgeThreshold;
}

void sift::set_sift_sigma(double th)
{
	sigma = th;
}

double sift::get_sift_sigma(void) const
{
	return sigma;
}

void sift::featuredetection(clipedmap_data& cm, target_data& td)
{

	cv::Ptr<cv::SIFT> detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

	detector->detectAndCompute(cm.oImage, cv::Mat(), cm.oPts, cm.oFeatures);
	detector->detectAndCompute(td.oImage, cv::Mat(), td.oPts, td.oFeatures);
}

