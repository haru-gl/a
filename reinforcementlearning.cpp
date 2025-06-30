#include "tmatrix.h"
#include "reinforcementlearning.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <math.h>

void disprawprop(const std::vector<double>& prop); 
void dispexpprop(const std::vector<double>& prop); 
void dispsqrtprop(const std::vector<double>& prop);
std::vector<size_t> softmax_y(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_y1(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_y2(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_y3(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_select(size_t numPts, const std::vector<double>& prop, size_t& sampleNum);
std::vector<size_t> softmax_org(size_t numPts, const std::vector<double>& prop, size_t& sampleNum, size_t cnt);
std::vector<size_t> softmax_n(size_t numPts, const std::vector<double>& prop, size_t sampleSize);
std::vector<size_t> softmax_n1(size_t numPts, const std::vector<double>& prop, size_t sampleSize,int& status);
std::vector<size_t> softmax_dc(size_t numPts, const std::vector<double>& disvalue, size_t sampleSize);

class InlierRateEstimator {
public:
    InlierRateEstimator() {
        // これらの係数は、いくつかのデータセットで事前に実験して決定する
        // predicted_rate = k * hist_stddev + c;
        k = 0.05; // 例: 標準偏差が大きいほどinlier率が高い
        c = 0.1;  // 例: ベースラインのinlier率
    }

    /**
     * @brief 画像の輝度ヒストグラムからinlier率を予測する
     * @param image 評価対象の画像 (dstd.oImage)
     * @return 予測されたinlier率 (0.0～1.0)
     */
    double predict(const cv::Mat& image) {
        if (image.empty() || image.channels() != 1) {
            return 0.5; // デフォルト値
        }

        // 1. 輝度ヒストグラムを計算
        cv::Mat hist;
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

        // 2. ヒストグラムの標準偏差を計算
        cv::Scalar mean, stddev;
        cv::meanStdDev(hist, mean, stddev);
        double hist_stddev = stddev[0];

        // 3. 簡単な線形回帰モデルでinlier率を予測
        double predicted_inlier_ratio = k * hist_stddev + c;

        // 予測値が物理的に意味のある範囲（例: 5%～90%）に収まるようにクリップ
        return std::max(0.05, std::min(0.90, predicted_inlier_ratio));
    }

private:
    double k, c; // 回帰モデルの係数
};

void reinforcementlearning::positionestimation_frfl(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_frfl(pe, cm, td, tform);
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

void reinforcementlearning::positionestimation_frfl_y(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_frfl_y(pe, cm, td, tform);
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

void reinforcementlearning::positionestimation_frfl_y1(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_frfl_y1(pe, cm, td, tform);
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

void reinforcementlearning::positionestimation_grfl(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_grfl(pe, cm, td, tform);
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

void reinforcementlearning::positionestimation_rfl_org(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_rfl_org(pe, cm, td, tform);
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

bool reinforcementlearning::sac_frfl(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = a_ransac_frfl(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = a_ransac_frfl_nd(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}

bool reinforcementlearning::sac_frfl_y(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = a_ransac_frfl_y(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = a_ransac_frfl_y(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}

bool reinforcementlearning::sac_frfl_y1(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = a_ransac_frfl_y1(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = a_ransac_frfl_y1(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}

bool reinforcementlearning::sac_grfl(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = a_ransac_grfl(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = a_ransac_grfl_nd(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    ////特徴点の様子を描画

    //cv::Mat results;
    //std::vector<cv::DMatch> matches;
    //std::vector<cv::KeyPoint> keypoints;
    //cv::drawMatches(td.oImage, keypoints, cm.oImage, keypoints, matches, results);
    //    for (int i = 0; i < selectedcm.size(); i++) {
    //        cv::line(results, cv::Point2d(selectedcm[i].x + SZ, selectedcm[i].y), cv::Point2d(selectedtd[i].x, selectedtd[i].y), cv::Scalar(0, 255, 255), 1);
    //}
    //
    //cv::imshow("sacimg", results);
    //cv::imwrite("sacimg_407_0878_mc.png", results);
    //cv::waitKey(0);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}

bool reinforcementlearning::sac_rfl_org(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = a_rfl_sac(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = a_rfl_sac(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}

double fuzzyE(double sdv, double x)
{
    return (sdv - x) / sdv;
}

double gaussE(double sdv, double x)
{
    return exp(-x * x / (2.0 * sdv * sdv));// / sqrt(2.0 * 3.14159265358979 * sdv * sdv);
}

double gaussm0E(double sdv, double x)
{
    return exp(-x * x / (0.5 * sdv * sdv));
}

double gaussm1E(double sdv, double x)
{
    return exp(-x / (0.5 * sdv));
}
double exE(double sdv, double x, double p)
{
    return 1.0 - pow(x / sdv, p);
}

double circle(double sdv, double x)
{
    return sdv - sqrt(sdv * sdv - (x - sdv) * (x - sdv)) ;
}
// ransac with reinforcemant learning
bool reinforcementlearning::a_ransac_frfl(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
    double maxvalue = 0.0;
    std::vector<double> prop(numPts), disvalue(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    double tsd;
    std::vector<double> exy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }
    for (int i = 0; i < numPts; i++) prop[i] = 0.0;

    for (it = 1; it <= maxIteration; it++) {
        indices = softmax(numPts, prop, sampleSize);

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                disvalue[j] = fuzzyE(tsd, norm);// (tsd - norm) / tsd;
                disvalue[j] = circle(tsd, norm);
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
        }

        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }

        for (int j = 0; j < numPts; j++) {
            int k;
            for (k = 0; k < sampleSize; k++)
                if (indices[k] == j) break;
            if (k != sampleSize) continue;
            prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
            
        }
        if (iterNum <= it) break;
    }
    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}
// ransac with reinforcemant learning
bool reinforcementlearning::a_ransac_frfl_y(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
    double maxvalue = 0.0;
    std::vector<double> prop(numPts), disvalue(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    double tsd;
    std::vector<double> exy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }

    std::random_device rnd;
    std::mt19937 mtr(rnd());
    std::uniform_real_distribution<> randp(0.45, 0.55);

    for (int i = 0; i < numPts; i++) prop[i] = randp(mtr);
//    for (int i = 0; i < numPts; i++) prop[i] = 0.0;

    for (it = 1; it <= maxIteration; it++) {
        indices = softmax_n(numPts, prop, sampleSize);

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                disvalue[j] = fuzzyE(tsd, norm);// (tsd - norm) / tsd;
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
        }

        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }
//        printf("sumvalue/maxvalue=%f\n", sumvalue / maxvalue);
        for (int j = 0; j < numPts; j++) {
            int k;
            for (k = 0; k < sampleSize; k++)
                if (indices[k] == j) break;
            if (k != sampleSize) continue;
//            prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
            prop[j] += alpha * (disvalue[j] - prop[j]);

        }
        if (iterNum <= it) break;
    }
    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

// ransac with reinforcemant learning
bool reinforcementlearning::a_ransac_frfl_y1(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);

    std::vector<double> prop(numPts), disvalue(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    double tsd;
    std::vector<double> exy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }
    for (int i = 0; i < numPts; i++) prop[i] = 0.0;
    double maxvalue = 0.0;
    size_t maxinlierNum = 0;
    int status;
    double p_tsd = tsd;
    double p_confidence = confidence=INIT_CONFIDENCE;
    double mconf = 100.0 - confidence;
    for (it = 1; it <= maxIteration; it++) {

        indices = softmax_n1(numPts, prop, sampleSize, status);
        switch (status) {
        case 4:
        case 5:
        case 6:
            tsd += 1.0;
            if (tsd > 15.0) tsd = 15.0;
            mconf *= 10.0; if (mconf > 0.9) mconf = 1.0;
            confidence = 100.0 - mconf;
            printf("************************************************************Set tsd=%f,conf=%f\n", tsd,confidence);
            break;
        }

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        //size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                //inlierNum++;
                inliersIdx[inlierNum++] = j;
                disvalue[j] = fuzzyE(tsd, norm);// (tsd - norm) / tsd;
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
            prop[j] += alpha * (disvalue[j] - prop[j]);
        }
        //sumvalue /= (double)inlierNum;
        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }
        if (inlierNum > maxinlierNum) {
            maxinlierNum = inlierNum;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }
        //        printf("sumvalue/maxvalue=%f\n", sumvalue / maxvalue);
//        for (int j = 0; j < numPts; j++) {
//            int k;
//            for (k = 0; k < sampleSize; k++)
//                if (indices[k] == j) break;
//            if (k != sampleSize) continue;
//            //            prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
//            prop[j] += alpha * (disvalue[j] - prop[j]);
//       }
        if (iterNum <= it) break;
    }
    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

// ransac with reinforcemant learning with normalization, denormalization(未完成)
bool reinforcementlearning::a_ransac_frfl_nd(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
    double maxvalue = 0.0;
    std::vector<double> prop(numPts), disvalue(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    std::vector<cv::Point2d> normalizedPts1(sampleSize);
    cv::Mat normalizedMat1;
    std::vector<cv::Point2d> normalizedPts2(sampleSize);
    cv::Mat normalizedMat2;
    double tsd;
    std::vector<double> exy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }
    for (int i = 0; i < numPts; i++) prop[i] = 0.0;

    for (it = 1; it <= maxIteration; it++) {
        indices = softmax(numPts, prop, sampleSize);

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }

        normalization(samplePts1, normalizedPts1, normalizedMat1);
        normalization(samplePts2, normalizedPts2, normalizedMat2);
        cv::Mat tform = computematrix(mt, normalizedPts1, normalizedPts2);
        denormalization(tform, normalizedMat1, normalizedMat2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                disvalue[j] = fuzzyE(tsd, norm); // (tsd - norm) / tsd;
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
        }
        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }
        for (int j = 0; j < numPts; j++) {
            int k;
            for(k=0;k<sampleSize;k++)
                if (indices[k] == j) break;
            if (k != sampleSize) continue;
            prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
        }
        if (iterNum <= it) break;
    }
    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    normalizedPts1.clear(); normalizedPts1.shrink_to_fit();
    normalizedPts2.clear(); normalizedPts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

bool reinforcementlearning::a_ransac_grfl(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;

    std::vector<size_t> indices = randperm(numPts, sampleSize);
    
    
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
    double maxvalue = 0.0;
    std::vector<double> prop(numPts), prop1(numPts), prop2(numPts), disvalue(numPts), disvalue1(numPts), disvalue2(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    double tsd;
    std::vector<double> exy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }
    for (int i = 0; i < numPts; i++) prop[i] = 0.0;

    // ===== 適応型パラメータ設定 =====
    InlierRateEstimator estimator;
    // dstd.oImageはどこかから参照する必要がある。ここでは仮に引数で渡すとする。
    // 実際には、この関数を呼び出すpositionestimation_normalから渡すのが良い。
    // double predicted_ratio = estimator.predict(dstd.oImage); 
    // ※ここでは仮にdstdを直接参照できないので、ダミー値で計算を進めます。
    // 実際にはdstd.oImageを渡してください。
    double predicted_ratio = estimator.predict(cv::Mat(100, 100, CV_8U, cv::Scalar(128))); // ダミー画像で予測

    // 1. maxIterationを適応的に設定
    size_t adaptive_iterNum = computeLoopNumbers(numPts, numPts * predicted_ratio, sampleSize);
    size_t execution_iterations = std::min((size_t)maxIteration, adaptive_iterNum);

    // 2. maxDistance (tsd) を適応的に設定
    const double base_maxDistance = th;
    const double scale_factor = 1.5; // 予測が悪いほど、しきい値をどのくらい緩和するかの係数
    double adaptive_maxDistance = base_maxDistance * (1.0 + (1.0 - predicted_ratio) * scale_factor);
    tsd = adaptive_maxDistance;

    printf("Adaptive RANSAC: Predicted Inlier Ratio=%.2f, Iterations=%zu, Threshold=%.2f\n",
        predicted_ratio, execution_iterations, tsd);

    for (it = 1; it <= maxIteration; it++) {
        indices = softmax_y(numPts, prop, sampleSize);
        

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        //td = 撮影画像, cm = 地図画像
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2); //変換に応じたサイズのベクトルで行列を作成

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++; //インライアの数のみ数える
                inliersIdx[iidx++] = j;
                //disvalue[j] = exE(tsd, norm, 0.965);
                disvalue[j] = fuzzyE(tsd, norm);
                //disvalue[j] = circle(tsd, norm);
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
        }

        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }
 
        for (int j = 0; j < numPts; j++) {
            int k;
            for (k = 0; k < sampleSize; k++)
                if (indices[k] == j) break;
            if (k != sampleSize) continue;
            
            //if (inlierNum > 4) prop[j] += sumvalue / maxvalue * alpha * (disvalue1[j] + 0.9 * prop1[j] - prop[j]);
            //else prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);

            prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
            
        }

        //if (inlierNum > 4)
        //{
        //    //TD(0)法の1step先の計算(予測)
        //    indices = softmax_y(numPts, prop, sampleSize);
        //    //for (size_t i = 0; i < 4; i++) // indices確認
        //    //    printf("indices = %zu\n" ,indices[i]); 

        //    for (int i = 0; i < sampleSize; i++) {
        //        samplePts1[i] = td[indices[i]];
        //        samplePts2[i] = cm[indices[i]];
        //    }
        //    //td = 撮影画像, cm = 地図画像
        //    cv::Mat tform1 = computematrix(mt, samplePts1, samplePts2); //変換に応じたサイズのベクトルで行列を作成

        //    for (size_t j = 0; j < numPts; j++) {
        //        cv::Point2d invPts = transform2d(td[j], tform1);
        //        cv::Point2d dist = invPts - cm[j];
        //        double norm = cv::norm(dist);
        //        if (norm < tsd) {
        //            disvalue1[j] = fuzzyE(tsd, norm);
        //        }
        //        else {
        //            norm = tsd;
        //            disvalue1[j] = 0.0;
        //        }
        //        prop1[j] = prop[j] + sumvalue / maxvalue * alpha * (disvalue1[j] - prop[j]);
        //    }
        //}
        if (iterNum <= it) break;
    }

 
    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

// ransac with reinforcemant learning with normalization, denormalization(未完成)
bool reinforcementlearning::a_ransac_grfl_nd(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
    double maxvalue = 0.0;
    std::vector<double> prop(numPts), disvalue(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    std::vector<cv::Point2d> normalizedPts1(sampleSize);
    cv::Mat normalizedMat1;
    std::vector<cv::Point2d> normalizedPts2(sampleSize);
    cv::Mat normalizedMat2;
    double tsd;
    std::vector<double> exy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }
    for (int i = 0; i < numPts; i++) prop[i] = 0.0;

    for (it = 1; it <= maxIteration; it++) {
        indices = softmax(numPts, prop, sampleSize);

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }

        normalization(samplePts1, normalizedPts1, normalizedMat1);
        normalization(samplePts2, normalizedPts2, normalizedMat2);
        cv::Mat tform = computematrix(mt, normalizedPts1, normalizedPts2);
        denormalization(tform, normalizedMat1, normalizedMat2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                disvalue[j] = gaussE(tsd, norm);// exp(-(tsd - norm) * (tsd - norm) / (2.0 * tsd * tsd));// (tsd - norm) / tsd;
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
        }
        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }
        for (int j = 0; j < numPts; j++) {
            int k;
            for(k=0;k<sampleSize;k++)
                if (indices[k] == j) break;
            if (k != sampleSize) continue;
            prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
        }
        if (iterNum <= it) break;
    }
    if (it == maxIteration || bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    normalizedPts1.clear(); normalizedPts1.shrink_to_fit();
    normalizedPts2.clear(); normalizedPts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

bool reinforcementlearning::a_rfl_sac(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t sampleNum = sampleSize;
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleNum);
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
    double maxvalue = 0.0;
    std::vector<double> prop(numPts), disvalue(numPts);// , pprop(numPts);
    std::vector<size_t> inliersIdx(numPts);
    size_t iterNum = ULLONG_MAX;
    std::vector<cv::Point2d> samplePts1(sampleNum), samplePts2(sampleNum);
    double tsd;
    std::vector<double> exy;

    switch (rm) {
    case ransacMode::dNORMAL:
        tsd = maxDistance;
        break;
    case ransacMode::dSTDDEV:
        tsd = th * stddv;
        break;
    case ransacMode::dHAMPLEI:
        tsd = th * medad * 1.4826;
        break;
    }
    for (int i = 0; i < numPts; i++) prop[i] = 0.0;

    for (it = 1; it <= maxIteration; it++) {
        sampleNum = sampleSize;
        indices = softmax_org(numPts, prop, sampleNum, it);

        for (int i = 0; i < sampleNum; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2);

        size_t inlierNum = 0;
        size_t iidx = 0;
        double sumvalue = 0.0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                disvalue[j] = exE(tsd, norm, 0.965);// gaussm1E(tsd, norm);
                exy.push_back(norm);
            }
            else {
                norm = tsd;
                disvalue[j] = 0.0;
            }
            sumvalue += disvalue[j];
        }

        //save best fit model & iterrationEvaluation
        if (sumvalue > maxvalue) {
            maxvalue = sumvalue;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
        }

        for (int j = 0; j < numPts; j++)
            prop[j] += alpha * (disvalue[j] - prop[j]);
        if (iterNum <= it) break;
//        if (it > 1000) {
//            double disprop = 0.0;
//            for (int j = 0; j < numPts; j++)
//                disprop += (prop[j] - pprop[j]) * (prop[j] - pprop[j]);
//            if (disprop < 1.0e-10) break;
//        }
//        for (int j = 0; j < numPts; j++) pprop[j] = prop[j];
    }
    if (it == maxIteration) return true;

    //bestInliersNum = sampleNum; std::cout << "best inliear num=" << sampleNum << std::endl;
    //bestInliersIdx = softmax_select(numPts, prop, bestInliersNum);
    std::cout << "best inliear num=" << bestInliersNum << std::endl;
    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[bestInliersIdx[i]]);
        selectedcm.push_back(cm[bestInliersIdx[i]]);
    }

    get_ave_stddev(exy, ave, stddv);
    med = get_median(exy);
    for (size_t j = 0; j < exy.size(); j++)
        exy[j] = fabs(exy[j] - med);
    medad = get_median(exy);

    bestInliersIdx.clear(); bestInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    prop.clear(); prop.shrink_to_fit();
    disvalue.clear(); disvalue.shrink_to_fit();
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}
