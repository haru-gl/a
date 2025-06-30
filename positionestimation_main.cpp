#include "classes.h"
#include "tmatrix.h"
#include "sac_main.h"
#include "reinforcementlearning.h"

//bool sac_svbayes(const matchingType mt, clipedmap_data& clpd, target_data& tgt, cv::Mat& tform);
//bool sac_svbayes_withknn(const matchingType mt, clipedmap_data& clpd, target_data& tgt, cv::Mat& tform);

//void featurematching_main(map_data& map, target_data& tgt, analysis_results& rst);
/*
void positionestimation_main(sac& sc, const matchingType mt, const ransacMode rm, const matrixcalType ct, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
	int minGoodPairs = get_minGP(mt); if (minGoodPairs == 0) { rs.status = 4; return; }
	if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
	else {
		cv::Mat tform;
		bool stat = true;
		stat = sc.sac_main(mt, rm, ct, cm, td, tform);
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
*/
/*
void positionestimation_rfl(reinforcementlearning& sc, const matchingType mt, const ransacMode rm, const matrixcalType ct,clipedmap_data& cm, target_data& td, analysis_results& rs)
{
	int minGoodPairs = get_minGP(mt); if (minGoodPairs == 0) { rs.status = 4; return; }
	if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
	else {
		cv::Mat tform;
		bool stat = true;
		stat = sc.sac_rfl(mt, rm, ct, cm, td, tform);
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
*/
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