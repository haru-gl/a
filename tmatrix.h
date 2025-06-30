#pragma once
#include <opencv2/opencv.hpp>
#include "enclasses.h"
#include "classes.h"

void get_ave_stddev(const std::vector<double>& exy, double& nave, double& nstddv);
double get_median(std::vector<double>& exy);

double cal_scale(const cv::Mat& tform, double szcenter);
cv::Point2d transform2d(const cv::Point2d pts, const cv::Mat& tform);
int get_minGP(const matchingType mt);
bool checkFunc(const cv::Mat& tform);
int get_xcol(const matchingType mt);
int get_yraw(const matchingType mt);
cv::Mat cnv_mt2vc(const matchingType mt, const cv::Mat& tform);
cv::Mat cnv_vc2mt(const matchingType mt, const cv::Mat& xn);
cv::Mat set_jacobian(const matchingType mt, std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2);
//double expellipse(const cv::Point2d& xy, const cv::Point2d& tsd);
std::vector<size_t> randperm(size_t size, size_t returnNum);
int solveZ_eigen(const cv::Mat& constraints, cv::Mat& u);//, double *err)
//std::vector<size_t> softmax(size_t numPts, const std::vector<double>& prop);
cv::Mat computematrix(const matchingType mt, std::vector<cv::Point2d>& tgpts1, std::vector<cv::Point2d>& cmpts2);
cv::Mat computematrix_byEigen(const matchingType mt, std::vector<cv::Point2d>& tgpts1, std::vector<cv::Point2d>& cmpts2);
void normalization(const std::vector<cv::Point2d>& inPts, std::vector<cv::Point2d>& normalizePts, cv::Mat& normalizeMat);
void denormalization(cv::Mat& tform, const cv::Mat& normMat1, const cv::Mat& normMat2);
bool map2template(const map_data& map, const target_data& target, clipedmap_data& clpd);
