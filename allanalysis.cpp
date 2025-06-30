#include <opencv2/opencv.hpp>
#include "classes.h"
#include "enclasses.h"
#include "allanalysis.h"

int get_minGP(const methods& ms);
double cal_scale(const cv::Mat& tform, double szcenter);
cv::Point2d transform2d(const cv::Point2d pts, const cv::Mat& tform);
bool map2template(const map_data& map, const target_data& target, clipedmap_data& clpd);
bool svbayes_main(sacAppType st, matchingType mt, clipedmap_data& clpd, target_data& tgt, cv::Mat& tform);

bool allanalysis::msac_main(clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
	std::vector<cv::Point2d> selectedcm, selectedtd;
	std::vector<cv::Point2d> selectedcm1, selectedtd1;
	std::vector<cv::Point2d> selectedcm2, selectedtd2;
	bool status = true, clf = true;
	switch (mms.st) {
	case sacAppType::sNORMAL:
		status = sct_sac.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm, selectedtd, sct_sac.maxDistance, true);
		if (status == false) status = sct_sac.matrixestimation(mms, selectedcm, selectedtd, tform);
		else {
			tform = cv::Mat();
			status = true;
		}
		break;
	case sacAppType::sRANSACNOR:
		clf = true;
		for (int i = 0; i < sct_nransac.get_num_ransac(); i++) {
			status = sct_nransac.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm, selectedtd, sct_nransac.maxDistance, clf);
			if (clf == true) clf = false;
		}
		if (status == false) status = sct_nransac.matrixestimation(mms, selectedcm, selectedtd, tform);
		else {
			tform = cv::Mat();
			status = true;
		}
		break;
	case sacAppType::sFILTERS1:
		status = sct_sd.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm1, selectedtd1, sct_sd.maxDistance, true);
		if (status == true) {
			tform = cv::Mat();
			status = true;
		}
		else {
			status = sct_sd.sactp_s(mms, ransacMode::dSTDDEV, selectedcm1, selectedtd1, selectedcm, selectedtd, sct_sd.th_sd, true);
			if (status == false) status = sct_sd.matrixestimation(mms, selectedcm, selectedtd, tform);
			else {
				tform = cv::Mat();
				status = true;
			}
		}
		break;
	case sacAppType::sFILTERS2:
		status = sct_sdsd.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm2, selectedtd2, sct_sdsd.maxDistance, true);
		if (status == true) {
			tform = cv::Mat();
			status = true;
		}
		else {
			status = sct_sdsd.sactp_s(mms, ransacMode::dSTDDEV, selectedcm2, selectedtd2, selectedcm1, selectedtd1, sct_sdsd.th1_sd, true);
			if (status == true) {
				tform = cv::Mat();
				status = true;
			}
			else {
				status = sct_sdsd.sactp_s(mms, ransacMode::dSTDDEV, selectedcm1, selectedtd1, selectedcm, selectedtd, sct_sdsd.th2_sd, true);
				if (status == false) status = sct_sdsd.matrixestimation(mms, selectedcm, selectedtd, tform);
				else {
					tform = cv::Mat();
					status = true;
				}
			}
		}
		break;
	case sacAppType::sFILTERH1:
		status = sct_hi.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm1, selectedtd1, sct_hi.maxDistance, true);
		if (status == true) {
			tform = cv::Mat();
			status = true;
		}
		else {
			status = sct_hi.sactp_s(mms, ransacMode::dHAMPLEI, selectedcm1, selectedtd1, selectedcm, selectedtd, sct_hi.th_hi, true);
			if (status == false) {
				status = sct_hi.matrixestimation(mms, selectedcm, selectedtd, tform);
			}
			else {
				tform = cv::Mat();
				status = true;
			}
		}
		break;
	case sacAppType::sFILTERH2:
		status = sct_hihi.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm2, selectedtd2, sct_hihi.maxDistance, true);
		if (status == true) {
			tform = cv::Mat();
			status = true;
		}
		else {
			status = sct_hihi.sactp_s(mms, ransacMode::dHAMPLEI, selectedcm2, selectedtd2, selectedcm1, selectedtd1, sct_hihi.th1_hi, true);
			if (status == true) {
				tform = cv::Mat();
				status = true;
			}
			else {
				status = sct_hihi.sactp_s(mms, ransacMode::dHAMPLEI, selectedcm1, selectedtd1, selectedcm, selectedtd, sct_hihi.th2_hi, true);
				if (status == false) status = sct_hihi.matrixestimation(mms, selectedcm, selectedtd, tform);
				else {
					tform = cv::Mat();
					status = true;
				}
			}
		}
		break;
	case sacAppType::sFILTERSH:
		status = sct_sdhi.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm2, selectedtd2, sct_sdhi.maxDistance, true);
		if (status == true) {
			tform = cv::Mat();
			status = true;
		}
		else {
			status = sct_sdhi.sactp_s(mms, ransacMode::dSTDDEV, selectedcm2, selectedtd2, selectedcm1, selectedtd1, sct_sdhi.th1_sd, true);
			if (status == true) {
				tform = cv::Mat();
				status = true;
			}
			else {
				status = sct_sdhi.sactp_s(mms, ransacMode::dHAMPLEI, selectedcm1, selectedtd1, selectedcm, selectedtd, sct_sdhi.th2_hi, true);
				if (status == false) status = sct_sdhi.matrixestimation(mms, selectedcm, selectedtd, tform);
				else {
					tform = cv::Mat();
					status = true;
				}
			}
		}
		break;
	case sacAppType::sFILTERHS:
		status = sct_hisd.sactp_m(mms, ransacMode::dNORMAL, cm, td, selectedcm2, selectedtd2, sct_hisd.maxDistance, true);
		if (status == true) {
			tform = cv::Mat();
			status = true;
		}
		else {
			status = sct_hisd.sactp_s(mms, ransacMode::dHAMPLEI, selectedcm2, selectedtd2, selectedcm1, selectedtd1, sct_hisd.th1_hi, true);
			if (status == true) {
				tform = cv::Mat();
				status = true;
			}
			else {
				status = sct_hisd.sactp_s(mms, ransacMode::dSTDDEV, selectedcm1, selectedtd1, selectedcm, selectedtd, sct_hisd.th2_sd, true);
				if (status == false) status = sct_hisd.matrixestimation(mms, selectedcm, selectedtd, tform);
				else {
					tform = cv::Mat();
					status = true;
				}
			}
		}
		break;
	}
	selectedcm.clear(); selectedcm.shrink_to_fit();
	selectedtd.clear(); selectedtd.shrink_to_fit();
	selectedcm1.clear(); selectedcm1.shrink_to_fit();
	selectedtd1.clear(); selectedtd1.shrink_to_fit();
	selectedcm2.clear(); selectedcm2.shrink_to_fit();
	selectedtd2.clear(); selectedtd2.shrink_to_fit();
	return status;
}

void allanalysis::positionestimation_main(clipedmap_data& cm, target_data& td, analysis_results& rs)
{
	int minGoodPairs = get_minGP(mms); if (minGoodPairs == 0) { rs.status = 4; return; }
	if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
	else {
		cv::Mat tform;
		bool stat = true;
		stat = allanalysis::msac_main(cm, td, tform);
		if (stat == false) {
			//			double scale_o = sqrt(tform.at<double>(0, 0) * tform.at<double>(0, 0) + tform.at<double>(1, 0) * tform.at<double>(1, 0));
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
		else rs.status = 2;//Matching not completed
	}
}

void allanalysis::positionestimation_vb(methods mms, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
	cv::Mat tform;
	bool stat = svbayes_main(mms.st, mms.mt, cm, td, tform);
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

void featurepoints_detection_main(const featureType &mt, clipedmap_data& clpd, target_data& tgt, analysis_results& rst)
{
	switch (mt) {
	case featureType::fAKAZE:
		{
			fdt_akaze.featuredetection(clpd, tgt);
			rst.map_ptsNum = (int)clpd.oPts.size();
			rst.target_ptsNum = (int)tgt.oPts.size();
		}
		break;
	case featureType::fKAZE:
		{
			fdt_kaze.featuredetection(clpd, tgt);
			rst.map_ptsNum = (int)clpd.oPts.size();
			rst.target_ptsNum = (int)tgt.oPts.size();
		}
		break;
	case featureType::fSURF:
		{
			fdt_surf.featuredetection(clpd, tgt);
			rst.map_ptsNum = (int)clpd.oPts.size();
			rst.target_ptsNum = (int)tgt.oPts.size();
		}
		break;
	case featureType::fSIFT:
		{
			fdt_sift.featuredetection(clpd, tgt);
			rst.map_ptsNum = (int)clpd.oPts.size();
			rst.target_ptsNum = (int)tgt.oPts.size();
		}
		break;
	case featureType::fBRISK:
		{
			fdt_brisk.featuredetection(clpd, tgt);
			rst.map_ptsNum = (int)clpd.oPts.size();
			rst.target_ptsNum = (int)tgt.oPts.size();
		}
		break;
	case featureType::fORB:
		{
			fdt_orb.featuredetection(clpd, tgt);
			rst.map_ptsNum = (int)clpd.oPts.size();
			rst.target_ptsNum = (int)tgt.oPts.size();
		}
		break;
	}
}

//特徴点検出法、変換行列形式の変更は列挙型定数の変更で行う
//分析の流れもこの関数の内容変更で行う
void allanalysis::featurematching_main(map_data& map, target_data& tgt, analysis_results& rst)
{
	//measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();
	//Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;
	rst.ms.ft = mms.ft;
	featurepoints_detection_main(clpd, tgt, rst);
//Correspondence point search by k-NN method
	rst.ms.kt = mms.kt;
	rst.goodPairsNum = knnt.match(mms.kt, clpd, tgt);
	md = std::chrono::system_clock::now();
	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();
	if (mms.st == sacAppType::sVBAYESONLY || mms.st==sacAppType::sVBAYESWITHKNN) {
		positionestimation_vb(mms, clpd, tgt, rst);
	}
	else {
		rst.ms.mt = mms.mt;// Shape of Transformation matrix
		rst.ms.ct = mms.ct;// Estimation method of transformation matrix
		rst.ms.rt = mms.rt;// Method of SAC
		rst.ms.st = mms.st;// Application method of SAC
		positionestimation_main(clpd, tgt, rst);
	}
	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

