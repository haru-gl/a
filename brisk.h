#pragma once
#include <opencv2/opencv.hpp>
#include "classes.h"

class brisk {
private:
	int brisk_Octaves;
	int brisk_Threshold;
public:
	brisk(void)
	{
		brisk_Octaves = INIT_BRISK_OCTAVES;//3
		brisk_Threshold = INIT_BRISK_THRESHOLD;//30.0
	}
	void set_brisk_Th(int th = INIT_BRISK_THRESHOLD);
	int get_brisk_Th(void) const;
	void set_brisk_parameters(cv::Ptr<cv::BRISK>& Detector);
	void featuredetection(clipedmap_data& cm, target_data& td);
};
