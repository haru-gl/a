#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "parameters.h"
#include "enclasses.h"
#include "sac_main.h"

class reinforcementlearning : public sac
{
public:
	double alpha;
	reinforcementlearning(void)
	{
		alpha = INIT_RL_ALPHA;					//for Reinforcement Learning
	}
	void set_rflearning_al(double v = INIT_RL_ALPHA)
	{
		alpha = v;
	}
	double get_rflearning_al(void) const
	{
		return alpha;
	}
	void positionestimation_frfl(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);
	void positionestimation_frfl_y(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);
	void positionestimation_frfl_y1(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);
	void positionestimation_grfl(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);
	void positionestimation_rfl_org(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs);
	bool sac_frfl(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);
	bool sac_grfl(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);
	bool sac_rfl_org(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);
	bool sac_frfl_y(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);
	bool sac_frfl_y1(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform);

	bool a_ransac_frfl(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	bool a_ransac_frfl_y(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	bool a_ransac_frfl_y1(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	bool a_ransac_frfl_nd(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	bool a_ransac_grfl(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	bool a_ransac_grfl_nd(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag);
	bool a_rfl_sac(const matchingType mt, const ransacMode rm, std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,	double th, bool clflag);

};
 
