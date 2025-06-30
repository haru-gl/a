#pragma once
#include <opencv2/opencv.hpp>
#include "classes.h"

class kaze {
private:
	double kaze_threshold;
	int kaze_nOctaves;
	int kaze_nOctaveLayers;
public:
	kaze(void)
	{
		kaze_threshold = INIT_KAZE_THRESHOLD;
		kaze_nOctaves = INIT_KAZE_NOCTAVES;
		kaze_nOctaveLayers = INIT_KAZE_NOCTARVELAYERS;
	}
	void set_parameters(cv::Ptr<cv::KAZE> &kazeDetector);
	void set_kazeTh(double th = INIT_KAZE_THRESHOLD);
	double get_kazeTh(void) const;
	void featuredetection(clipedmap_data& cm, target_data& td);
};

