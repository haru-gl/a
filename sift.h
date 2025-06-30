#pragma once
#include <opencv2/opencv.hpp>
#include "classes.h"

class sift {
private:
	int nfeatures;
	int nOctaveLayers;
	double contrastThreshold;
	double edgeThreshold;
	double sigma;
public:
	sift(void)
	{
		nfeatures = INIT_SIFT_NFEATURES;// 0;
		nOctaveLayers = INIT_SIFT_NOCTAVELAYERS;// 3;
		contrastThreshold = INIT_SIFT_CONTRASTTH;// 0.04;
		edgeThreshold = INIT_SIFT_EDGETH;// 10;
		sigma = INIT_SIFT_SIGMA;// 1.6;
	}
	void set_sift_contrastTh(double th = INIT_SIFT_CONTRASTTH);
	double get_sift_contrastTh(void) const;
	void set_sift_egdeTh(double th = INIT_SIFT_EDGETH);
	double get_sift_edgeTh(void) const;
	void set_sift_sigma(double th = INIT_SIFT_SIGMA);
	double get_sift_sigma(void) const;
	void featuredetection(clipedmap_data& cm, target_data& td);
};
