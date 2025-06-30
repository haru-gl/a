#include <opencv2/opencv.hpp>
#include "akaze.h"
//#include "classes.h"

bool map2template(const map_data& map, const target_data& target, clipedmap_data& clpd);

//特徴点検出法、変換行列形式の変更は列挙型定数の変更で行う
//分析の流れもこの関数の内容変更で行う
void featurematching_main(map_data& map, target_data& tgt, analysis_results& rst)
{
	//measurement of time
	std::chrono::system_clock::time_point st, md, ed;
	st = std::chrono::system_clock::now();

	//Cutting out map images based on GNC data
	clipedmap_data clpd;
	if (!map2template(map, tgt, clpd)) return;

////featurepoints_detection(clpd, tgt, rst);/////
	akaze fdt_akaze;
//	fdt_akaze.set_akazeTh(double th);
//	fdt_akaze.set_parameters(cv::Ptr<cv::AKAZE> &detector);
	fdt_akaze.featuredetection(clpd, tgt);
	rst.map_ptsNum = (int)clpd.oPts.size();
	rst.target_ptsNum = (int)tgt.oPts.size();


	//Correspondence point search by k-NN method
	rst.goodPairsNum = knn_matching(clpd, tgt);

	md = std::chrono::system_clock::now();
	long long comtime = std::chrono::duration_cast<std::chrono::milliseconds>(md - st).count();

	positionestimation(clpd, tgt, rst);

	ed = std::chrono::system_clock::now();
	rst.elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ed - md).count() + comtime;
}

/*
void allanalysis::featurepoints_detection_main(clipedmap_data& clpd, target_data& tgt, analysis_results& rst)
{
	switch (mms.ft) {
	case featureType::fAKAZE:
		fdt_akaze.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
		break;
	case featureType::fKAZE:
		fdt_kaze.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
		break;
	case featureType::fSURF:
		fdt_surf.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
		break;
	case featureType::fSIFT:
		fdt_sift.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
		break;
	case featureType::fBRISK:
		fdt_brisk.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
		break;
	case featureType::fORB:
		fdt_orb.featuredetection(clpd, tgt);
		rst.map_ptsNum = (int)clpd.oPts.size();
		rst.target_ptsNum = (int)tgt.oPts.size();
		break;
	}
}
*/
/*
size_t knn::match(knnType kt, clipedmap_data& cmp, target_data& tgt)
{
	//	k-NN
	cv::BFMatcher matcher;
	const bool isCrossCheck = false;
	matcher = cv::BFMatcher(knn_normType, isCrossCheck);
	std::vector<std::vector<cv::DMatch>> nn_matches;
	matcher.knnMatch(tgt.oFeatures, cmp.oFeatures, nn_matches, knn_k);

	size_t goodPairsNum = 0;
	size_t candnum = 0;
	std::vector<cv::Point2d> cmppts, tgtpts;
	cv::Mat cmpdes, tgtdes;
	std::vector<float> ratio;
	size_t rejt = 0, rejm = 0;
	size_t rfixn;

	cmp.oMatchedPts.clear(); cmp.oMatchedFeatures = cv::Mat(); tgt.oMatchedPts.clear(); tgt.oMatchedFeatures = cv::Mat();
	cmppts.clear(); cmpdes = cv::Mat(); tgtpts.clear(); tgtdes = cv::Mat();
	switch (kt) {
	case knnType::kNORMAL:
		goodPairsNum = matchiratiocheck(nn_matches, cmp, tgt, cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio);

		if (knn_sort) kp_sort(cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio);
		break;
	case knnType::kDTSFP://Delete the correspondance points that use the same feature point
		candnum = matchiratiocheck(nn_matches, cmp, tgt, cmppts, cmpdes, tgtpts, tgtdes, ratio);

		//同じ特徴点を利用した対応点を削除する
		rejt = detect_samepoint(tgtpts, ratio);
		rejm = detect_samepoint(cmppts, ratio);
		std::cout << "number of rejected points=" << rejt << "+" << rejm << std::endl;

		goodPairsNum = 0;
		for (size_t n = 0; n < candnum; n++) {
			if (ratio[n] >= 1.0) continue;
			goodPairsNum++;
			tgt.oMatchedPts.push_back(tgtpts[n]);
			tgt.oMatchedFeatures.push_back(tgtdes.row((int)n));
			cmp.oMatchedPts.push_back(cmppts[n]);
			cmp.oMatchedFeatures.push_back(cmpdes.row((int)n));
		}

		if (knn_sort) kp_sort(cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio);
		break;
	case knnType::kNNFIXN:
		candnum = matchiratiocheck(nn_matches, cmp, tgt, cmppts, cmpdes, tgtpts, tgtdes, ratio);

		//Matiratioの小さいfixn個の対応点を選択する
		goodPairsNum = 0;
		rfixn = knn_fixn;
		if (rfixn > candnum) rfixn = candnum;
		for (size_t n = 0; n < rfixn; n++) {
			if (ratio[n] >= 1.0f) continue;
			float minratio = 1.0f;
			size_t mini = 0;
			for (size_t i = 0; i < candnum; i++) {
				if (minratio > ratio[i]) {
					minratio = ratio[i];
					mini = i;
				}
			}
			goodPairsNum++;
			tgt.oMatchedPts.push_back(tgtpts[mini]);
			tgt.oMatchedFeatures.push_back(tgtdes.row((int)mini));
			cmp.oMatchedPts.push_back(cmppts[mini]);
			cmp.oMatchedFeatures.push_back(cmpdes.row((int)mini));
			ratio[mini] = 1.1f;
		}

		if (knn_sort) kp_sort(cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio);
		break;

	case knnType::kNNFIXNDTSFP:
		candnum = matchiratiocheck(nn_matches, cmp, tgt, cmppts, cmpdes, tgtpts, tgtdes, ratio);

		//同じ特徴点を利用した対応点を削除する
		rejt = detect_samepoint(tgtpts, ratio);
		rejm = detect_samepoint(cmppts, ratio);
		std::cout << "number of rejected points=" << rejt << "+" << rejm << std::endl;

		//Matiratioの小さいfixn個の対応点を選択する
		goodPairsNum = 0;
		rfixn = knn_fixn;
		if (rfixn > candnum) rfixn = candnum;
		for (size_t n = 0; n < rfixn; n++) {
			if (ratio[n] >= 1.0f) continue;
			float minratio = 1.0f;
			size_t mini = 0;
			for (size_t i = 0; i < candnum; i++) {
				if (minratio > ratio[i]) {
					minratio = ratio[i];
					mini = i;
				}
			}
			goodPairsNum++;
			tgt.oMatchedPts.push_back(tgtpts[mini]);
			tgt.oMatchedFeatures.push_back(tgtdes.row((int)mini));
			cmp.oMatchedPts.push_back(cmppts[mini]);
			cmp.oMatchedFeatures.push_back(cmpdes.row((int)mini));
			ratio[mini] = 1.1f;
		}

		if (knn_sort) kp_sort(cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio);
		break;
	}
	tgtpts.clear(); tgtpts.shrink_to_fit();
	cmppts.clear(); cmppts.shrink_to_fit();
	ratio.clear(); ratio.shrink_to_fit();
	tgtdes = cv::Mat(); cmpdes = cv::Mat();

	nn_matches.clear();
	nn_matches.shrink_to_fit();
	std::cout << "goodPairsNum=" << goodPairsNum << std::endl;
	return goodPairsNum;
}
*/

size_t knn::knn_matching(clipedmap_data& cmp, target_data& tgt)
{
	//	k-NN
	cv::BFMatcher matcher;
	const bool isCrossCheck = false;
	matcher = cv::BFMatcher(knn_normType, isCrossCheck);
	std::vector<std::vector<cv::DMatch>> nn_matches;
	matcher.knnMatch(tgt.oFeatures, cmp.oFeatures, nn_matches, knn_k);

	size_t goodPairsNum = 0;
	size_t candnum = 0;
	std::vector<cv::Point2d> cmppts, tgtpts;
	cv::Mat cmpdes, tgtdes;
	std::vector<float> ratio;

	cmp.oMatchedPts.clear(); cmp.oMatchedFeatures = cv::Mat(); tgt.oMatchedPts.clear(); tgt.oMatchedFeatures = cv::Mat();
	cmppts.clear(); cmpdes = cv::Mat(); tgtpts.clear(); tgtdes = cv::Mat();
	goodPairsNum = matchiratiocheck(nn_matches, cmp, tgt, cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio);

//		if (knn_sort) kp_sort(cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio);
	tgtpts.clear(); tgtpts.shrink_to_fit();
	cmppts.clear(); cmppts.shrink_to_fit();
	ratio.clear(); ratio.shrink_to_fit();
	tgtdes = cv::Mat(); cmpdes = cv::Mat();

	nn_matches.clear();
	nn_matches.shrink_to_fit();
	std::cout << "goodPairsNum=" << goodPairsNum << std::endl;
	return goodPairsNum;
}
*/