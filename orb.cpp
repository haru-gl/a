#include <opencv2/opencv.hpp>
#include "orb.h"

void orb::set_orb_maxfeatures(int mf)
{
	orb_maxfeatures = mf;
}

int orb::get_orb_maxfeatures(void) const
{
	return orb_maxfeatures;
}

void orb::set_orb_scalefactor(double mf)
{
	orb_scalefactor = mf;
}

double orb::get_orb_scalefactor(void) const
{
	return orb_scalefactor;
}

void orb::set_orb_edgethreshold(int mf)
{
	orb_edgethreshold = mf;
}

int orb::get_orb_edgethreshold(void) const
{
	return orb_edgethreshold;
}

void orb::set_orb_fastthreshold(int mf)
{
	orb_fastthreshold = mf;
}

int orb::get_orb_fastthreshold(void) const
{
	return orb_fastthreshold;
}

void orb::set_parameters(cv::Ptr<cv::ORB>& detector)
{
	detector->setMaxFeatures(orb_maxfeatures);		//500:The maximum number of features to retain
	detector->setScaleFactor(orb_scalefactor);		//1.2f:Pyramid decimation ratio, greater than 1.
	detector->setNLevels(3);					//3:The number of pyramid levels.
	detector->setEdgeThreshold(orb_edgethreshold);	//31:Size of the border where the features are not detected.
	detector->setFirstLevel(0);					//0:It should be 0.
	detector->setWTA_K(2);						//
	detector->setScoreType(cv::ORB::HARRIS_SCORE);//HARRIS_SCORE, FAST_SCORE
	detector->setPatchSize(31);					//31:Size of the patch used by the oriented BRIEF descriptor.
	detector->setFastThreshold(orb_fastthreshold);	//20:
}

void orb::featuredetection(clipedmap_data& cm, target_data& td)
{
	cv::Ptr<cv::ORB> detector = cv::ORB::create();
	set_parameters(detector);
	detector->detectAndCompute(cm.oImage, cv::Mat(), cm.oPts, cm.oFeatures);
	detector->detectAndCompute(td.oImage, cv::Mat(), td.oPts, td.oFeatures);

}
