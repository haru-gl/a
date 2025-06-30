#pragma once
#include <opencv2/opencv.hpp>
#include "classes.h"

class surf{
private:
	double surf_Threshold;
public:
	surf(void)
	{
		surf_Threshold = INIT_SURF_THRESHOLD;
	}
	void set_surfTh(double th = INIT_SURF_THRESHOLD);
	double get_surfTh(void) const;
	bool extended = false; //false 64ŽŸŒ³  true 128ŽŸŒ³
	void featuredetection(clipedmap_data& cm, target_data& td);
};
