#include <opencv2/opencv.hpp>
#include "classes.h"
#include "gaussnewton.h"

cv::Mat set_jacobian(const matchingType mt, std::vector<cv::Point2d>& samplePts1, std::vector<cv::Point2d>& samplePts2);
//cv::Mat invMChoresky(const cv::Mat& a);

cv::Mat invL(const cv::Mat& l)
{
	cv::Mat il = cv::Mat::eye(l.size(), CV_64FC1);

	for (int i = 0; i < il.cols; i++) {
		double d = l.at<double>(i, i);
		for (int j = 0; j < il.rows; j++)
			il.at<double>(i, j) /= d;
		for (int k = i + 1; k < il.cols; k++) {
			double e = l.at<double>(k, i);
			for (int j = 0; j < k; j++)
				il.at<double>(k, j) -= e * il.at<double>(i, j);
		}
	}
	return il;
}

cv::Mat invD(const cv::Mat& d)
{
	cv::Mat id = cv::Mat::zeros(d.size(), CV_64FC1);
	for (int i = 0; i < id.cols; i++)
		id.at<double>(i, i) = 1.0 / d.at<double>(i, i);
	return id;
}

void MChoreskyL(const cv::Mat& a, cv::Mat& d, cv::Mat& l)
{
	d = cv::Mat::zeros(a.size(), CV_64FC1);
	l = cv::Mat::eye(a.size(), CV_64FC1);
	for (int k = 0; k < a.cols; k++) {
		double sum = 0.0;
		for (int j = 0; j < k; j++)
			sum += l.at<double>(k, j) * l.at<double>(k, j) * d.at<double>(j, j);
		d.at<double>(k, k) = a.at<double>(k, k) - sum;
		for (int i = k + 1; i < a.cols; i++) {
			sum = 0.0;
			for (int j = 0; j < k; j++)
				sum += l.at<double>(i, j) * d.at<double>(j, j) * l.at<double>(k, j);
			l.at<double>(i, k) = (a.at<double>(i, k) - sum) / d.at<double>(k, k);
		}
	}
}

cv::Mat invMChoresky(const cv::Mat& a)
{
	cv::Mat d, l;
	MChoreskyL(a, d, l);
	cv::Mat id = invD(d);
	cv::Mat il = invL(l);
	cv::Mat ilt; cv::transpose(il, ilt);
	l = ilt * id;
	cv::Mat ia = l * il;
	return ia;
}

// Gauss-Newton Method
cv::Mat gaussnewtonmethod::gauss_newton(const matchingType mt, std::vector<cv::Point2d>& selectedtd, std::vector<cv::Point2d>& selectedcm, cv::Mat& x0)
{
	size_t numPts = selectedtd.size();
	cv::Mat h = x0.clone(), xn = x0.clone();

	std::vector<cv::Point2d> samplePts1(numPts), samplePts2(numPts);
	for (int i = 0; i < numPts; i++) {
		samplePts1[i] = selectedtd[i];
		samplePts2[i] = selectedcm[i];
	}
	cv::Mat Jx = set_jacobian(mt, samplePts1, samplePts2);
	cv::Mat Jxt; cv::transpose(Jx, Jxt);
	cv::Mat H = Jxt * Jx;
	cv::Mat invH = invMChoresky(H);

	for (size_t n = 0; n < maxitr; n++) {
		cv::Mat e = Jx * xn;
		double abs_e = cv::norm(e);
		std::cout << " |e|=" << abs_e << std::endl;
		if (abs_e < eps) break;
		cv::Mat g = Jxt * e;
		h = -invH * g;
		xn = xn + alp * h;
	}
	return xn;
}

