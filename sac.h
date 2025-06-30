#pragma once
#include <opencv2/opencv.hpp>
//#include "classes.h"
#include "parameters.h"
#include "gaussnewton.h"
#include "reinforcementlearning.h"
#include "kerneldensityestimation.h"

class sac : public gaussnewtonmethod, public reinforcementlearning, public kerneldensityestimation {
public:
	int maxIteration;				//åJÇËï‘ÇµÇÃç≈ëÂêî
	double confidence;				//ämìxÅì
	double maxDistance;				//ç≈ëÂãóó£
    double ave,stddv;               //Average and standard deviation of the cordinate
    double med,medad;               //Median and Median absoute deviation
    double csd;
    
    sac(void) {
		maxIteration = INIT_MAXITERATION;	//Basic ransac
		confidence = INIT_CONFIDENCE;		//Basic ransac
		maxDistance = INIT_MAXDISTANCE;		//Basic ransac
        ave = stddv = med = medad= 0.0;
        csd = 1.0;
	}
	void set_maxIteration(int th = INIT_MAXITERATION);
	int get_maxIteration(void) const;
	void set_confidence(double th = INIT_CONFIDENCE);
	double get_confidence(void) const;
	void set_maxDistance(double th = INIT_MAXDISTANCE);
	double get_maxDistance(void) const;

	size_t computeLoopNumbers(size_t numPts, size_t inlierNum, size_t sampleSize);
	bool matrixestimation(const methods& ms, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform);
    bool sactp_s(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cmall, std::vector<cv::Point2d>& tdall, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = 1.0, bool clflag = true);
    bool sactp_m(const methods& ms, const ransacMode rm, clipedmap_data& cm, target_data& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);

    bool a_ransac(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = 1.0, bool clflag = true);
    bool a_ransac_nd(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = 1.0, bool clflag = true);
    bool a_ransac_rfl(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = 1.0, bool clflag = true);
    bool a_ransac_rfl_nd(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = 1.0, bool clflag = true);
    bool a_kde(const methods& ms, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = INIT_DISTERROR, bool clflag = true);
    bool a_kde_nd(const methods& ms, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = INIT_DISTERROR, bool clflag = true);
    bool a_prosac(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = 1.0, bool clflag = true);
    bool a_prosac_nd(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th = 1.0, bool clflag = true);
 //   bool vbayes_dcp(const methods& ms, clipedmap_data& clpd, target_data& tgt, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd);
};
