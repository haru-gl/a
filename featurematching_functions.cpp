#include "classes.h"
#include "enclasses.h"
#include "tmatrix.h"
#include "knn.h"
#include "sac_main.h"
#include "reinforcementlearning.h"
#include "featurematching_config.h" // InitialMatcherType enum
#include "custom_lsh_matcher.h"   // é©çÏLSH
#include "flann_lsh_matcher.h"    // FLANN LSH
#include <chrono>
#include <iostream>

void featurepointsdetection(const featureType& ft, clipedmap_data& clpd, target_data& tgt, analysis_results& rst);

void featurematching_normal0(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_normal1(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_normal1_dr(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_rfl0(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_rfl1(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_rfl2(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_taubin0(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_dr(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_rfl_org(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_rfl_y(map_data& map, target_data& tgt, analysis_results& rst);
void featurematching_rfl_y1(map_data& map, target_data& tgt, analysis_results& rst);

void featurematching_main(map_data& map, target_data& tgt, analysis_results& rst)
{
//	featurematching_normal0(map, tgt, rst);
//	featurematching_normal1(map, tgt, rst);
//	featurematching_rfl0(map, tgt, rst);
//	featurematching_rfl1(map, tgt, rst);
	featurematching_rfl2(map, tgt, rst);
//	featurematching_taubin0(map, tgt, rst);
//	featurematching_dr(map, tgt, rst);
//	featurematching_normal1_dr(map, tgt, rst);
//	featurematching_rfl_org(map, tgt, rst);
//	featurematching_rfl_y(map, tgt, rst);
//	featurematching_rfl_y1(map, tgt, rst);

}

void featurematching_dr(map_data& map, target_data& tgt, analysis_results& rst)
{
	//Setting of Feature Points Detection
	featureType ft = featureType::fAKAZE;
	//Setting of Position Estimation
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	//Setting of kNN
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	sac sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_dr(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_normal0(map_data& map, target_data& tgt, analysis_results& rst)
{
	//Setting of Feature Points Detection
	featureType ft = featureType::fAKAZE;
	//Setting of Position Estimation
	posestType pe;
	pe.mt = matchingType::mSIMILARITY;
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	//Setting of kNN
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	sac sc;

//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
//	Position Estimation
	sc.positionestimation_normal(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_normal1(map_data& map, target_data& tgt, analysis_results& rst)
{
	//Setting of Feature Points Detection
	featureType ft = featureType::fAKAZE;
	//Setting of Position Estimation
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;//********************
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	//Setting of kNN
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	knnt.set_knn_k(4);
	sac sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_normal(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_normal1_dr(map_data& map, target_data& tgt, analysis_results& rst)
{
	//Setting of Feature Points Detection
	featureType ft = featureType::fAKAZE;
	//Setting of Position Estimation
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;//********************
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	//Setting of kNN
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	knnt.set_knn_k(4);
	sac sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_normal_dr(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_rfl_y(map_data& map, target_data& tgt, analysis_results& rst)
{
	featureType ft = featureType::fAKAZE;
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	reinforcementlearning sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_frfl_y(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_rfl_y1(map_data& map, target_data& tgt, analysis_results& rst)
{
	featureType ft = featureType::fAKAZE;
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE; //mSIMILARITY;// mPROJECTIVE;
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	reinforcementlearning sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_frfl_y1(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_rfl0(map_data& map, target_data& tgt, analysis_results& rst)
{
	featureType ft = featureType::fAKAZE;
	posestType pe;
	pe.mt = matchingType::mSIMILARITY;
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	reinforcementlearning sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_frfl(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_rfl1(map_data& map, target_data& tgt, analysis_results& rst)
{
	featureType ft = featureType::fAKAZE;
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;//******************
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	reinforcementlearning sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_frfl(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_rfl2(map_data& map, target_data& tgt, analysis_results& rst)
{
	featureType ft = featureType::fAKAZE; //fSURF fAKAZE
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;//******************
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	reinforcementlearning sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_grfl(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_rfl_org(map_data& map, target_data& tgt, analysis_results& rst)
{
	featureType ft = featureType::fAKAZE;
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;//******************
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cSVD;
	pe.ndon = false;
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	reinforcementlearning sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_rfl_org(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

void featurematching_taubin0(map_data& map, target_data& tgt, analysis_results& rst)
{
	featureType ft = featureType::fAKAZE;
	posestType pe;
	pe.mt = matchingType::mPROJECTIVE;
	pe.rm = ransacMode::dNORMAL;
	pe.ct = matrixcalType::cTAUBIN;
	pe.ndon = false;
	knnType kt = knnType::kNORMAL;
	knn knnt; knnt.set_knn_sortflag(false);
	//reinforcementlearning sc;
	sac sc;

	//	Measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//	Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

	//	Feature point detection
	featurepointsdetection(ft, clpd, tgt, rst);

	//	Correspondence point search by k-NN method
	rst.goodPairsNum = knnt.match(kt, clpd, tgt);
	md = std::chrono::system_clock::now();

	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	//	Position Estimation
	sc.positionestimation_normal(pe, clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

/*
void positionestimation_svbayes(const matchingType mt, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
	cv::Mat tform;
	bool stat = sac_svbayes(mt, cm, td, tform);
	rs.scale = cal_scale(tform, td.szcenter);
	cv::Point2d estimated;
	estimated = transform2d(cv::Point2d(td.szcenter, td.szcenter), tform);
	rs.estimatedCenter2dx = estimated.x + cm.lux;
	rs.estimatedCenter2dy = estimated.y + cm.luy;
	rs.estimatedHeight = cm.averageDistance * rs.scale + cm.averageHeight - 600.0;//The meaning of "600" is unknown
	rs.c00 = transform2d(cv::Point2d(0.0, 0.0), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.c01 = transform2d(cv::Point2d((double)td.sz, 0.0), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.c11 = transform2d(cv::Point2d((double)td.sz, (double)td.sz), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.c10 = transform2d(cv::Point2d(0.0, (double)td.sz), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.status = 0;
}

void positionestimation_svbayes_withknn(const matchingType mt, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
	cv::Mat tform;
	bool stat = sac_svbayes_withknn(mt, cm, td, tform);
	rs.scale = cal_scale(tform, td.szcenter);
	cv::Point2d estimated;
	estimated = transform2d(cv::Point2d(td.szcenter, td.szcenter), tform);
	rs.estimatedCenter2dx = estimated.x + cm.lux;
	rs.estimatedCenter2dy = estimated.y + cm.luy;
	rs.estimatedHeight = cm.averageDistance * rs.scale + cm.averageHeight - 600.0;//The meaning of "600" is unknown
	rs.c00 = transform2d(cv::Point2d(0.0, 0.0), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.c01 = transform2d(cv::Point2d((double)td.sz, 0.0), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.c11 = transform2d(cv::Point2d((double)td.sz, (double)td.sz), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.c10 = transform2d(cv::Point2d(0.0, (double)td.sz), tform) + cv::Point2d(cm.lux, cm.luy);
	rs.status = 0;
}
*/
