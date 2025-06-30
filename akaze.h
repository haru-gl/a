#pragma once
#include <opencv2/opencv.hpp>
#include "classes.h"

class akaze{
private:
	int akaze_descriptor_size;
	int akaze_descriptor_channels;
	double akaze_threshold;
	int akaze_nOctaves;
	int akaze_nOctaveLayers;
public:
	akaze(void)
	{
		akaze_descriptor_size = INIT_AKAZE_DESCRIPTOR_SIZE;
		akaze_descriptor_channels = INIT_AKAZE_DESCRIPTOR_CHANNELS;
		akaze_threshold = INIT_AKAZE_THRESHOLD;
		akaze_nOctaves = INIT_AKAZE_NOCTAVES;
		akaze_nOctaveLayers = INIT_AKAZE_NOCTARVELAYERS;
	}
	void set_parameters(cv::Ptr<cv::AKAZE> &detector);
	void set_akazeTh(double th = INIT_AKAZE_THRESHOLD);
	double get_akazeTh(void) const;
	void featuredetection(clipedmap_data& cm, target_data& td);
};

