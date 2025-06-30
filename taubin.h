#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

cv::Mat set_ks(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2, int a, int k);
cv::Mat set_T(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2, int a, int k);
Eigen::MatrixXd CvM2EgM(cv::Mat& X);
cv::Mat EgM2CvM(Eigen::MatrixXd& A);
cv::Mat Get_v(cv::Mat& M, cv::Mat& N);
cv::Mat GenInv(cv::Mat& X);
cv::Mat compute_x0(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2);
cv::Mat hyper_renormalization(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2, cv::Mat& x0);

