#pragma once
#include "parameters.h"
#include "enclasses.h"

class gaussnewtonmethod
{
public:
	double alp, bet, eps;
	size_t maxitr;
	gaussnewtonmethod(void)
	{
		alp = INIT_GN_ALPHA;
		bet = INIT_GN_BETA;
		eps = INIT_GN_EPS;
		maxitr = INIT_GN_MAXITR;
	}
	cv::Mat gauss_newton(const matchingType mt, std::vector<cv::Point2d>& selectedtd, std::vector<cv::Point2d>& selectedcm, cv::Mat& x0);

	void set_gn_alpha(double v = INIT_GN_ALPHA)
	{
		alp = v;
	}
	double get_gn_alpha(void) const
	{
		return alp;
	}
	void set_gn_beta(double v = INIT_GN_BETA)
	{
		bet = v;
	}
	double get_gn_beta(void) const
	{
		return bet;
	}
	void set_gn_epsiron(double v = INIT_GN_EPS)
	{
		eps = v;
	}
	double get_gn_epsiron(void) const
	{
		return eps;
	}
	void set_gn_maxiter(size_t v = INIT_GN_MAXITR)
	{
		maxitr = v;
	}
	size_t get_gn_maxiter(void) const
	{
		return maxitr;
	}
};
