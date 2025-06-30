#pragma once
#include <random>
#include <opencv2/opencv.hpp>
#include "parameters.h"
//#include "enclasses.h"

class map_data {
public:	//	処理する画像サイズ
	//マップデータ
	int lsz;						//特徴点の探索範囲(1024)
	double averageDistance;			//マップ作成時の想定高度における、平均地表からの距離[m]
	double averageHeight;			//マップ領域の平均標高[m]
	cv::Mat oImage;					//マップ画像

	map_data(void)
	{
		lsz = INIT_LSZ;
		averageDistance = averageHeight = 0.0;
		oImage = cv::Mat();
	}
	~map_data(void)
	{
		oImage = cv::Mat();
	}
};

class clipedmap_data {
public:
	int lux, luy;
	double averageDistance;			//Average Distance of the map data
	double averageHeight;			//Average Height of the map data
	cv::Mat oImage;
	std::vector<cv::KeyPoint> oPts;
	cv::Mat oFeatures;
	std::vector<cv::Point2d> oMatchedPts;
	cv::Mat oMatchedFeatures;

	clipedmap_data(void)
	{
		lux = luy = 0;
		oImage = cv::Mat();
		oPts.clear();
		oFeatures = cv::Mat();
		oMatchedPts.clear();
		oMatchedFeatures = cv::Mat();
	}
	~clipedmap_data(void)
	{
		oImage = cv::Mat();
		oPts.clear(); oPts.shrink_to_fit();
		oFeatures = cv::Mat();
		oMatchedPts.clear(); oMatchedPts.shrink_to_fit();
		oMatchedFeatures = cv::Mat();
	}
};

class target_data {
public:
	int sz;						//Size of Target image・sz x sz)
	double szcenter;			//Center cordinate of the Target Image
	double x_gnc, y_gnc;		//GNC Data
	cv::Mat oImage;				//Image Data
	std::vector<cv::KeyPoint> oPts;
	cv::Mat oFeatures;
	std::vector<cv::Point2d> oMatchedPts;
	cv::Mat oMatchedFeatures;

	target_data(void)
	{
		sz = SZ; szcenter = SZCENTER;
		x_gnc = y_gnc = INFINITY;
		oImage = cv::Mat();
		oPts.clear();
		oFeatures = cv::Mat();
		oMatchedPts.clear();
		oMatchedFeatures = cv::Mat();
	}
	~target_data(void)
	{
		oImage = cv::Mat();
		oPts.clear(); oPts.shrink_to_fit();
		oFeatures = cv::Mat();
		oMatchedPts.clear(); oMatchedPts.shrink_to_fit();
		oMatchedFeatures = cv::Mat();
	}
	void dd_clear(void)
	{
		oPts.clear(); oPts.shrink_to_fit();
		oFeatures = cv::Mat();
		oMatchedPts.clear(); oMatchedPts.shrink_to_fit();
		oMatchedFeatures = cv::Mat();
	}
};

class analysis_results {
public:
	double estimatedCenter2dx, estimatedCenter2dy;	//推定座標
	double scale;									//推定スケール
	double estimatedHeight;							//推定高度
	long long elapsedTime;							//処理時間
	size_t map_ptsNum, target_ptsNum;				//検出特徴点数
	size_t goodPairsNum;							//有効対応点数
	int status;										//状態：0:正検出、2:マッチング未了、3:インライア不足、4：その他))
	cv::Point2d c00, c01, c11, c10;					//地図における撮影画像の四隅の座標（左上c00、右上c01、右下c11、左下c10）

	analysis_results(void)
	{
		estimatedCenter2dx = estimatedCenter2dy = scale = estimatedHeight = INFINITY;
		elapsedTime = 0;
		map_ptsNum = target_ptsNum = goodPairsNum = status = 0;
		c00 = c01 = c11 = c10 = cv::Point2d(0.0, 0.0);
	}
};

class simulation_data {
public:
	double x_true, y_true, h_true;		//画像データベースによる真値
	double distTh;						//正検出とする誤差(ピクセル)
	double errDx, errDy, errD;			//真値との距離
	double errHeightP;					//高度誤差
	simulation_data(void)
	{
		x_true = y_true = h_true = INFINITY;
		distTh=	INIT_DISTTH;
		errDx = errDy = errD = errHeightP = INFINITY;
	}
};

class statistics {
public:
	int count0, count1, count2, count3;
	double sc0dmean, sc0dmax, sc0dmin, sc0d3sg;
	double sc0dxmean, sc0dxmax, sc0dxmin, sc0dx3sg;
	double sc0dymean, sc0dymax, sc0dymin, sc0dy3sg;
	double sc0hgmean, sc0hgmax, sc0hgmin, sc0hg3sg;
	double sc0tmean, sc0tmax, sc0tmin, sc0t3sg;
	double sc01dmean, sc01dmax, sc01dmin, sc01d3sg;
	double sc01dxmean, sc01dxmax, sc01dxmin, sc01dx3sg;
	double sc01dymean, sc01dymax, sc01dymin, sc01dy3sg;
	double sc01hgmean, sc01hgmax, sc01hgmin, sc01hg3sg;
	double sc01tmean, sc01tmax, sc01tmin, sc01t3sg;
	double altmean, altmax, altmin, alt3sg;

	statistics(void) {
		count0 = count1 = count2 = count3 = 0;
		sc0dmean = 0.0; sc0dmax = -INFINITY; sc0dmin = INFINITY; sc0d3sg = 0.0;
		sc0dxmean = 0.0; sc0dxmax = -INFINITY; sc0dxmin = INFINITY; sc0dx3sg = 0.0;
		sc0dymean = 0.0; sc0dymax = -INFINITY; sc0dymin = INFINITY; sc0dy3sg = 0.0;
		sc0hgmean = 0.0; sc0hgmax = -INFINITY; sc0hgmin = INFINITY; sc0hg3sg = 0.0;
		sc0tmean = 0.0; sc0tmax = -INFINITY; sc0tmin = INFINITY; sc0t3sg = 0.0;
		sc01dmean = 0.0; sc01dmax = -INFINITY; sc01dmin = INFINITY; sc01d3sg = 0.0;
		sc01dxmean = 0.0; sc01dxmax = -INFINITY; sc01dxmin = INFINITY; sc01dx3sg = 0.0;
		sc01dymean = 0.0; sc01dymax = -INFINITY; sc01dymin = INFINITY; sc01dy3sg = 0.0;
		sc01hgmean = 0.0; sc01hgmax = -INFINITY; sc01hgmin = INFINITY; sc01hg3sg = 0.0;
		sc01tmean = 0.0; sc01tmax = -INFINITY; sc01tmin = INFINITY; sc01t3sg = 0.0;
		altmean = 0.0; altmax = -INFINITY; altmin = INFINITY; alt3sg = 0.0;
	}
	void clear(void)
	{
		count0 = count1 = count2 = count3 = 0;
		sc0dmean = 0.0; sc0dmax = -INFINITY; sc0dmin = INFINITY; sc0d3sg = 0.0;
		sc0dxmean = 0.0; sc0dxmax = -INFINITY; sc0dxmin = INFINITY; sc0dx3sg = 0.0;
		sc0dymean = 0.0; sc0dymax = -INFINITY; sc0dymin = INFINITY; sc0dy3sg = 0.0;
		sc0hgmean = 0.0; sc0hgmax = -INFINITY; sc0hgmin = INFINITY; sc0hg3sg = 0.0;
		sc0tmean = 0.0; sc0tmax = -INFINITY; sc0tmin = INFINITY; sc0t3sg = 0.0;
		sc01dmean = 0.0; sc01dmax = -INFINITY; sc01dmin = INFINITY; sc01d3sg = 0.0;
		sc01dxmean = 0.0; sc01dxmax = -INFINITY; sc01dxmin = INFINITY; sc01dx3sg = 0.0;
		sc01dymean = 0.0; sc01dymax = -INFINITY; sc01dymin = INFINITY; sc01dy3sg = 0.0;
		sc01hgmean = 0.0; sc01hgmax = -INFINITY; sc01hgmin = INFINITY; sc01hg3sg = 0.0;
		sc01tmean = 0.0; sc01tmax = -INFINITY; sc01tmin = INFINITY; sc01t3sg = 0.0;
		altmean = 0.0; altmax = -INFINITY; altmin = INFINITY; alt3sg = 0.0;
	}
	void add_stat(const analysis_results& rst, const simulation_data& simd);
	void cal_stat(void);
};

std::ostream& operator<<(std::ostream& os, const map_data& md);
std::ostream& operator<<(std::ostream& os, const clipedmap_data& md);
std::ostream& operator<<(std::ostream& os, const target_data& md);
std::ostream& operator<<(std::ostream& os, const analysis_results& md);
std::ostream& operator<<(std::ostream& os, const simulation_data& md);
std::ostream& operator<<(std::ostream& os, const statistics& md);
std::ofstream& operator<<(std::ofstream& os, const statistics& md);
