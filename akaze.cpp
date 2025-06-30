#include <opencv2/opencv.hpp>
#include "akaze.h"

void akaze::set_akazeTh(double th)
{
	akaze_threshold = th;
}

double akaze::get_akazeTh(void) const
{
	return akaze_threshold;
}

void akaze::set_parameters(cv::Ptr<cv::AKAZE> &detector)
{
	detector->setDescriptorType(cv::AKAZE::DESCRIPTOR_MLDB);
	detector->setDiffusivity(cv::KAZE::DIFF_PM_G2);
	detector->setThreshold(akaze_threshold);
	detector->setDescriptorSize(akaze_descriptor_size);
	detector->setDescriptorChannels(akaze_descriptor_channels);
	detector->setNOctaves(akaze_nOctaves);
	detector->setNOctaveLayers(akaze_nOctaveLayers);
}

void akaze::featuredetection(clipedmap_data& cm, target_data& td)
{
	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
	set_parameters(detector);
	detector->detectAndCompute(cm.oImage, cv::Mat(), cm.oPts, cm.oFeatures);
	std::cout << "Row=" << cm.oFeatures.rows << ",Col=" << cm.oFeatures.cols << ",type=" << cm.oFeatures.type() << std::endl;
	detector->detectAndCompute(td.oImage, cv::Mat(), td.oPts, td.oFeatures);
}

//Mat type()
//        C1  C2  C3  C4
// CV_8U   0   8  16  24
// CV_8S   1   9  17  25
// CV_16U  2  10  18  26
// CV_16S  3  11  19  27
// CV_32S  4  12  20  28
// CV_32F  5  13  21  29
// CV_64F  6  14  22  30
