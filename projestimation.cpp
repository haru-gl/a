#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/opencv.hpp>
#include "enclasses.h"

Eigen::VectorXd projmatrixestimation(const matchingType mt, std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2)
{
	int cl = (int)p1.size();
	Eigen::MatrixXd nn(9, 9);
	Eigen::MatrixXd mm(9, 9);
	Eigen::VectorXd rt(9);
	double dcl = (double)cl;
	double f0 = 512.0, f2 = f0 * f0;

	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 9; j++)
			nn(i, j) = mm(i, j) = 0.0;
		rt(i) = 0.0;
	}

	if (mt != matchingType::mPROJECTIVE_EV) return rt;
	for (int i = 0; i < cl; i++) {
		double x2 = p1[i].x * p1[i].x, y2 = p1[i].y * p1[i].y;
		double xd2 = p2[i].x * p2[i].x, yd2 = p2[i].y * p2[i].y;
		double xy = p1[i].x * p1[i].y, fx = f0 * p1[i].x, fy = f0 * p1[i].y;
		double mxdyd = -p2[i].x * p2[i].y;
		double mfxd = -f0 * p2[i].x, mfyd = -f0 * p2[i].y;
		double f2yd2 = f2 + yd2, f2xd2 = f2 + xd2;
		double xd2yd2 = xd2 + yd2;

		nn(0, 0) += x2 + f2yd2;
		nn(0, 1) += xy;
		nn(0, 2) += fx;
		nn(0, 3) += mxdyd;
		nn(0, 4) += 0.0;
		nn(0, 5) += 0.0;
		nn(0, 6) += mfxd;
		nn(0, 7) += 0.0;
		nn(0, 8) += 0.0;

		nn(1, 0) += xy;
		nn(1, 1) += y2 + f2yd2;
		nn(1, 2) += fy;
		nn(1, 3) += 0.0;
		nn(1, 4) += mxdyd;
		nn(1, 5) += 0.0;
		nn(1, 6) += 0.0;
		nn(1, 7) += mfxd;
		nn(1, 8) += 0.0;

		nn(2, 0) += fx;
		nn(2, 1) += fy;
		nn(2, 2) += f2;
		nn(2, 3) += 0.0;
		nn(2, 4) += 0.0;
		nn(2, 5) += 0.0;
		nn(2, 6) += 0.0;
		nn(2, 7) += 0.0;
		nn(2, 8) += 0.0;

		nn(3, 0) += mxdyd;
		nn(3, 1) += 0.0;
		nn(3, 2) += 0.0;
		nn(3, 3) += x2 + f2xd2;
		nn(3, 4) += xy;
		nn(3, 5) += fx;
		nn(3, 6) += mfyd;
		nn(3, 7) += 0.0;
		nn(3, 8) += 0.0;

		nn(4, 0) += 0.0;
		nn(4, 1) += mxdyd;
		nn(4, 2) += 0.0;
		nn(4, 3) += xy;
		nn(4, 4) += y2 + f2xd2;
		nn(4, 5) += fy;
		nn(4, 6) += 0.0;
		nn(4, 7) += mfyd;
		nn(4, 8) += 0.0;

		nn(5, 0) += 0.0;
		nn(5, 1) += 0.0;
		nn(5, 2) += 0.0;
		nn(5, 3) += fx;
		nn(5, 4) += fy;
		nn(5, 5) += f2;
		nn(5, 6) += 0.0;
		nn(5, 7) += 0.0;
		nn(5, 8) += 0.0;

		nn(6, 0) += mfxd;
		nn(6, 1) += 0.0;
		nn(6, 2) += 0.0;
		nn(6, 3) += mfyd;
		nn(6, 4) += 0.0;
		nn(6, 5) += 0.0;
		nn(6, 6) += x2 + xd2 + yd2;
		nn(6, 7) += 2.0 * xy;
		nn(6, 8) += 2.0 * fx;

		nn(7, 0) += 0.0;
		nn(7, 1) += mfxd;
		nn(7, 2) += 0.0;
		nn(7, 3) += 0.0;
		nn(7, 4) += mfxd;
		nn(7, 5) += 0.0;
		nn(7, 6) += 2.0 * xy;
		nn(7, 7) += y2 + xd2 + yd2;
		nn(7, 8) += 2.0 * fy;

		nn(8, 0) += 0.0;
		nn(8, 1) += 0.0;
		nn(8, 2) += 0.0;
		nn(8, 3) += 0.0;
		nn(8, 4) += 0.0;
		nn(8, 5) += 0.0;
		nn(8, 6) += 2.0 * fx;
		nn(8, 7) += 2.0 * fy;
		nn(8, 8) += 2.0 * f2;

		mm(0, 0) += x2 * f2yd2;
		mm(0, 1) += xy * f2yd2;
		mm(0, 2) += fx * f2yd2;
		mm(0, 3) += x2 * mxdyd;
		mm(0, 4) += xy * mxdyd;
		mm(0, 5) += fx * mxdyd;
		mm(0, 6) += x2 * mfxd;
		mm(0, 7) += xy * mfxd;
		mm(0, 8) += fx * mfxd;

		mm(1, 0) += xy * f2yd2;
		mm(1, 1) += y2 * f2yd2;
		mm(1, 2) += fy * f2yd2;
		mm(1, 3) += xy * mxdyd;
		mm(1, 4) += y2 * mxdyd;
		mm(1, 5) += fy * mxdyd;
		mm(1, 6) += xy * mfxd;
		mm(1, 7) += y2 * mfxd;
		mm(1, 8) += fy * mfxd;

		mm(2, 0) += fx * f2yd2;
		mm(2, 1) += fy * f2yd2;
		mm(2, 2) += f2 * f2yd2;
		mm(2, 3) += fx * mxdyd;
		mm(2, 4) += fy * mxdyd;
		mm(2, 5) += f2 * mxdyd;
		mm(2, 6) += fx * mfxd;
		mm(2, 7) += fy * mfxd;
		mm(2, 8) += f2 * mfxd;

		mm(3, 0) += x2 * mxdyd;
		mm(3, 1) += xy * mxdyd;
		mm(3, 2) += fx * mxdyd;
		mm(3, 3) += x2 * f2xd2;
		mm(3, 4) += xy * f2xd2;
		mm(3, 5) += fx * f2xd2;
		mm(3, 6) += x2 * mfyd;
		mm(3, 7) += xy * mfyd;
		mm(3, 8) += fx * mfyd;

		mm(4, 0) += xy * mxdyd;//論文の式(2)の間違い
		mm(4, 1) += y2 * mxdyd;//論文の式(2)の間違い
		mm(4, 2) += fy * mxdyd;//論文の式(2)の間違い
		mm(4, 3) += xy * f2xd2;
		mm(4, 4) += y2 * f2xd2;
		mm(4, 5) += fy * f2xd2;
		mm(4, 6) += xy * mfyd;
		mm(4, 7) += y2 * mfyd;
		mm(4, 8) += fy * mfyd;

		mm(5, 0) += fx * mxdyd;
		mm(5, 1) += fy * mxdyd;
		mm(5, 2) += f2 * mxdyd;
		mm(5, 3) += fx * f2xd2;
		mm(5, 4) += fy * f2xd2;
		mm(5, 5) += f2 * f2xd2;
		mm(5, 6) += fx * mfyd;
		mm(5, 7) += fy * mfyd;
		mm(5, 8) += f2 * mfyd;

		mm(6, 0) += x2 * mfxd;
		mm(6, 1) += xy * mfxd;
		mm(6, 2) += fx * mfxd;
		mm(6, 3) += x2 * mfyd;
		mm(6, 4) += xy * mfyd;
		mm(6, 5) += fx * mfyd;
		mm(6, 6) += x2 * xd2yd2;
		mm(6, 7) += xy * xd2yd2;
		mm(6, 8) += fx * xd2yd2;
		
		mm(7, 0) += xy * mfxd;
		mm(7, 1) += y2 * mfxd;
		mm(7, 2) += fy * mfxd;
		mm(7, 3) += xy * mfyd;
		mm(7, 4) += y2 * mfyd;
		mm(7, 5) += fy * mfyd;
		mm(7, 6) += xy * xd2yd2;
		mm(7, 7) += y2 * xd2yd2;
		mm(7, 8) += fy * xd2yd2;
		
		mm(8, 0) += fx * mfxd;
		mm(8, 1) += fy * mfxd;
		mm(8, 2) += f2 * mfxd;
		mm(8, 3) += fx * mfyd;
		mm(8, 4) += fy * mfyd;
		mm(8, 5) += f2 * mfyd;
		mm(8, 6) += fx * xd2yd2;
		mm(8, 7) += fy * xd2yd2;
		mm(8, 8) += f2 * xd2yd2;
	}
	for (int i = 0; i < 9; i++)
		for (int j = 0; j < 9; j++) {
			nn(i, j) /= dcl;
			mm(i, j) /= dcl;
		}
	Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(nn, mm);
	Eigen::VectorXd evl = es.eigenvalues();
	Eigen::MatrixXd evc = es.eigenvectors();
	std::cout << evl << std::endl;
	std::cout << evc << std::endl;
	//最大固有値に対応する固有ベクトルのみを使う
	//evのなかにどのように保存されているのかわからないので、この先は保留です。
	//最終的に必要なのは絶対値最大の固有値に対応した固有ベクトル１つなので、パワー法のようなものでできないか検討します。
	return evl;
}
