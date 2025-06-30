#pragma once
#include <opencv2/opencv.hpp>
#include "classes.h"

class orb {
private:
	int orb_maxfeatures;// = INIT_ORBMAXFEATURES;//4000
	double orb_scalefactor;// = INIT_ORBSCALEFACTOR;//1.2f
	int orb_edgethreshold;// = INIT_ORBTHRESHOLD;//31
	int orb_fastthreshold;// = INITORBFASTTHRESHOLD;//20
public:
	orb(void)
	{
		orb_maxfeatures= INIT_ORB_MAXFEATURES;//4000
		orb_scalefactor = INIT_ORB_SCALEFACTOR;//1.2f
		orb_edgethreshold = INIT_ORB_THRESHOLD;//31
		orb_fastthreshold = INIT_ORB_FASTTHRESHOLD;//20
	}
	void set_orb_maxfeatures(int mf = INIT_ORB_MAXFEATURES);
	int get_orb_maxfeatures(void) const;
	void set_orb_scalefactor(double mf = INIT_ORB_SCALEFACTOR);
	double get_orb_scalefactor(void) const;
	void set_orb_edgethreshold(int mf = INIT_ORB_THRESHOLD);
	int get_orb_edgethreshold(void) const;
	void set_orb_fastthreshold(int mf = INIT_ORB_FASTTHRESHOLD);
	int get_orb_fastthreshold(void) const;
	void set_parameters(cv::Ptr<cv::ORB>& Detector);
	void featuredetection(clipedmap_data& cm, target_data& td);
};
