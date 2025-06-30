#include "tmatrix.h"
#include "sac_main.h"
#include "taubin.h"

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

double cal_area(const std::vector<cv::Point2d>& s, int o, int a, int b)
{
    return 0.5 * fabs((s[a].x - s[o].x) * (s[b].y - s[o].y) - (s[a].y - s[o].y) * (s[b].x - s[o].x));
}

double double_rate(const std::vector<cv::Point2d>& s)
{
    //△OAB
    double oab1 = cal_area(s, 0, 1, 2);// 0.5 * fabs((s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[1].y - s[0].y) * (s[2].x - s[0].x));
    //△OAC
    double oac1 = cal_area(s, 0, 1, 3);// 0.5 * fabs((s[1].x - s[0].x) * (s[3].y - s[0].y) - (s[1].y - s[0].y) * (s[3].x - s[0].x));
    //△OCD
    double ocd1 = cal_area(s, 0, 3, 4);// 0.5 * fabs((s[3].x - s[0].x) * (s[4].y - s[0].y) - (s[3].y - s[0].y) * (s[4].x - s[0].x));
    //△OBD
    double obd1 = cal_area(s, 0, 2, 4);// 0.5 * fabs((s[2].x - s[0].x) * (s[4].y - s[0].y) - (s[2].y - s[0].y) * (s[4].x - s[0].x));
    return (oab1 * ocd1) / (oac1 * obd1);
}

double eudist(const cv::Point2d& a, const cv::Point2d& b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void lgdort(std::vector<size_t>& d)
{
    int nm = d.size();
    for (int i = 0; i < nm - 1; i++)
        for(int j = i + 1 ; j < nm; j++)
            if (d[i] < d[j]) {
                size_t tmp = d[i];
                d[i] = d[j];
                d[j] = tmp;
            }
}

size_t inliertuning_dr(const std::vector<cv::Point2d>& cmoMatchedPts, const std::vector<cv::Point2d>& tdoMatchedPts, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd)
{
    size_t tnum = selectedtd.size();
    size_t otnum = tdoMatchedPts.size();
    double cnerr = 5.0;
    const size_t sampleSize = 5;
    std::vector<size_t> indices = randperm(tnum, sampleSize);
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);

    for (size_t it = 0; it < tnum; it++) {
        //std::cout << "it=" << it << ", Number of fps=" << selectedtd.size() << "::";
        if (selectedtd.size() < sampleSize) break;
        indices = randperm(selectedtd.size(), sampleSize);
        lgdort(indices);
        for (int i = 0; i < sampleSize; i++)
            samplePts2[i] = selectedcm[indices[i]];
        double dr_cm = double_rate(samplePts2);
        if (fabs(dr_cm) < 0.1) continue;//複比が小さすぎる場合評価を避ける

        std::vector<size_t> drc[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            drc[i].clear();
            for (size_t j = 0; j < otnum; j++) {
                double r = eudist(selectedtd[indices[i]], tdoMatchedPts[j]);
                if (r <= cnerr) drc[i].push_back(j);
            }
        }

        int canall = 0;
        for (int i = 0; i < sampleSize; i++)
            canall += drc[i].size();
        if (canall == sampleSize) continue;

        int tt[sampleSize] = { 0 };
        double minerr = 100000.0;
        for (int t0 = 0; t0 < drc[0].size(); t0++) {
            samplePts1[0] = tdoMatchedPts[drc[0][t0]];
            for (int t1 = 0; t1 < drc[1].size(); t1++) {
                samplePts1[1] = tdoMatchedPts[drc[1][t1]];
                for (int t2 = 0; t2 < drc[2].size(); t2++) {
                    samplePts1[2] = tdoMatchedPts[drc[2][t2]];
                    for (int t3 = 0; t3 < drc[3].size(); t3++) {
                        samplePts1[3] = tdoMatchedPts[drc[3][t3]];
                        for (int t4 = 0; t4 < drc[4].size(); t4++) {
                            samplePts1[4] = tdoMatchedPts[drc[4][t4]];
                            double dr_td = double_rate(samplePts1);
                            double err = fabs((dr_td - dr_cm) / dr_cm);
                            if (minerr > err) {
                                minerr = err;
                                tt[0] = t0; tt[1] = t1; tt[2] = t2; tt[3] = t3; tt[4] = t4;
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < sampleSize; i++) {
            if (eudist(selectedtd[indices[i]], tdoMatchedPts[drc[i][tt[i]]]) > 0.1) {
                //printf("td[%d] is deleted\n", (int)indices[i]);
                selectedtd.erase(selectedtd.begin() + indices[i]);//元の対応点以外に複比誤差の小さいものがあった場合、その対応点を削除
                selectedcm.erase(selectedcm.begin() + indices[i]);//地図側の対応点も削除
            }
            //selectedtd[indices[i]] = tdoMatchedPts[drc[i][tt[i]]];//複比誤差が最も小さい特徴点を採用する場合
        }
    }
    std::cout << "Num after DR tuning=" << selectedtd.size() << std::endl;
    return selectedtd.size();
}

/* 未完成
void sac::positionestimation_drransac(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_drransac(pe, cm, td, tform);
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
void sac::positionestimation_dr(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_dr(pe, cm, td, tform);
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

void sac::positionestimation_normal(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_normal(pe, cm, td, tform);
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

void sac::positionestimation_normal_dr(const posestType& pe, clipedmap_data& cm, target_data& td, analysis_results& rs)
{
    int minGoodPairs = get_minGP(pe.mt); if (minGoodPairs == 0) { rs.status = 4; return; }
    if (rs.goodPairsNum < minGoodPairs) rs.status = 3;//Insufficient inlayer
    else {
        cv::Mat tform;
        bool stat = true;
        stat = sac_normal_dr(pe, cm, td, tform);
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

bool sac::sac_dr(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    status = sac_dr_only(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, 0.005, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}
/*
bool sac::sac_drransac(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = dr_ransac(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = dr_ransac_nd(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}
*/
bool sac::sac_normal(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = a_ransac(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = a_ransac_nd(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}

bool sac::sac_normal_dr(const posestType& pe, clipedmap_data& cm, target_data& td, cv::Mat& tform)
{
    std::vector<cv::Point2d> selectedcm, selectedtd;
    bool status = true;

    if (pe.ndon == false) status = a_ransac(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    else status = a_ransac_nd(pe.mt, pe.rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, maxDistance, true);
    if (status == false) 
        inliertuning_dr(cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd);

    if (status == false) status = matrixestimation(pe.mt, pe.ct, selectedcm, selectedtd, tform);
    else {
        tform = cv::Mat();
        status = true;
    }
    return status;
}

bool sac::matrixestimation(const matchingType mt, const matrixcalType ct, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform)
{
    bool status = true;
    cv::Mat x0, xn1;
    switch (ct) {
    case matrixcalType::cSVD:
        tform = computematrix(mt, selectedtd, selectedcm);
        break;
    case matrixcalType::cSVD_EIGEN:
        tform = computematrix_byEigen(mt, selectedtd, selectedcm);
        break;
    case matrixcalType::cGAUSSNEWTON:
        tform = computematrix(mt, selectedtd, selectedcm);
        status = checkFunc(tform);
        x0 = cnv_mt2vc(mt, tform);
        xn1 = gauss_newton(mt, selectedtd, selectedcm, x0);
        tform = cnv_vc2mt(mt, xn1);
        break;
    case matrixcalType::cGAUSSNEWTON_EIGEN:
        tform = computematrix_byEigen(mt, selectedtd, selectedcm);
        status = checkFunc(tform);
        x0 = cnv_mt2vc(mt, tform);
        xn1 = gauss_newton(mt, selectedtd, selectedcm, x0);
        tform = cnv_vc2mt(mt, xn1);
        break;
    case matrixcalType::cTAUBIN:
        if (mt != matchingType::mPROJECTIVE && mt != matchingType::mPROJECTIVE3) {
            std::cout << "MatcingType is not Projective" << std::endl;
            tform = cv::Mat();
            break;
        }
        x0 = compute_x0(selectedtd, selectedcm);
        xn1 = hyper_renormalization(selectedtd, selectedcm, x0);
        tform = cnv_vc2mt(mt, xn1);
        break;
    }
    status = checkFunc(tform);
    return status;
}

void sac::set_maxIteration(size_t th)
{
    maxIteration = th;
}

int sac::get_maxIteration(void) const
{
    return maxIteration;
}

void sac::set_confidence(double th)
{
    confidence = th;
}

double sac::get_confidence(void) const
{
    return confidence;
}

void sac::set_maxDistance(double th)
{
    maxDistance = th;
}

double sac::get_maxDistance(void) const
{
    return maxDistance;
}

size_t sac::computeLoopNumbers(size_t numPts, size_t inlierNum, size_t sampleSize)
{
    double eps = 1.0e-15;
    double inlierProbability = 1.0;//initial value=1
    size_t nn;
    double factor = (double)inlierNum / numPts;
    for (size_t i = 0; i < sampleSize; i++)
        inlierProbability *= factor;

    if (inlierProbability < eps) nn = INT_MAX;
    else {
        double conf = confidence / 100.0;
        double numerator = log10(1 - conf);
        double denominator = log10(1 - inlierProbability);
        nn = (size_t)(numerator / denominator);
    }
    return nn;
}


// double rate ransac
bool sac::sac_dr_only(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = 5;// (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (it = 1; it <= maxIteration; it++) {
        indices = randperm(numPts, sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        double dr_td = double_rate(samplePts1);
        double dr_cm = double_rate(samplePts2);
        double err = fabs(dr_td - dr_cm) / dr_cm * 100.0;
        if (err < th) {
            for (int i = 0; i < sampleSize; i++) {
                selectedtd.push_back(td[indices[i]]);
                selectedcm.push_back(cm[indices[i]]);
            }
            bestInliersNum += sampleSize;
         }
    }
    std::cout << "Number of Inliers detected by Double Rate=" << bestInliersNum << std::endl;
    if (bestInliersNum < sampleSize) return true;

    indices.clear(); indices.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();

    return false;
}

bool sac::dr_ransac(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    double minErr = DBL_MAX;
    std::vector<size_t> bestInliersIdx(numPts);
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

    for (it = 1; it <= maxIteration; it++) {
        indices = randperm(numPts, sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        double err = 0.0;
        size_t iidx = 0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                exy.push_back(norm);
            }
            else
                norm = maxDistance;
            err += norm;
        }
        //save best fit model & iterrationEvaluation
        if (err < minErr) {
            minErr = err;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
            if (iterNum <= it) break;
        }
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
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

// basic ransac
bool sac::a_ransac(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    size_t it, bestInliersNum = 0;
    double minErr = DBL_MAX;
    std::vector<size_t> bestInliersIdx(numPts);
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

    for (it = 1; it <= maxIteration; it++) {
        indices = randperm(numPts, sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(mt, samplePts1, samplePts2);

        //model evaluation(M-Estimator)
        size_t inlierNum = 0;
        double err = 0.0;
        size_t iidx = 0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                exy.push_back(norm);
            }
            else
                norm = maxDistance;
            err += norm;
        }
        //save best fit model & iterrationEvaluation
        if (err < minErr) {
            minErr = err;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
            if (iterNum <= it) break;
        }
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
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}

// ransac with normalization and denormalization
bool sac::a_ransac_nd(const matchingType mt, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(mt);
    size_t numPts = td.size(); if (numPts < sampleSize) return true;
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    double minErr = DBL_MAX;
    size_t it, bestInliersNum = 0;
    std::vector<size_t> bestInliersIdx(numPts);
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
    for (it = 1; it <= maxIteration; it++) {
        indices = randperm(numPts, sampleSize);

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
        double err = 0.0;
        size_t iidx = 0;
        for (size_t j = 0; j < numPts; j++) {
            //calculate transform distance
            cv::Point2d invPts = transform2d(td[j], tform);
            cv::Point2d dist = invPts - cm[j];
            double norm = cv::norm(dist);
            if (norm < tsd) {
                inlierNum++;
                inliersIdx[iidx++] = j;
                exy.push_back(norm);
            }
            else
                norm = maxDistance;
            err += norm;
        }
        //save best fit model & iterrationEvaluation
        if (err < minErr) {
            minErr = err;
            bestInliersNum = inlierNum;
            bestInliersIdx = inliersIdx;
            iterNum = computeLoopNumbers(numPts, inlierNum, sampleSize);
            if (iterNum <= it) break;
        }
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
    inliersIdx.clear(); inliersIdx.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    exy.clear(); exy.shrink_to_fit();

    return false;
}


/*
bool sac::sactp_m(const methods& ms, const ransacMode rm, clipedmap_data& cm, target_data& td, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag)
{
    bool status = true;
    switch (ms.rt) {
    case sacType::rRANSAC:
        status = a_ransac(ms, rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rRANSACWITHNORM:
        status = a_ransac_nd(ms, rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rREINFORCEMENT:
        status = a_ransac_rfl(ms, rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rREINFORCEMENTWITHNORM:
        status = a_ransac_rfl_nd(ms, rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rKERNELDE:
        status = a_kde(ms, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rKERNELDEWITHNORM:
        status = a_kde_nd(ms, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rPROSAC:
        status = a_prosac(ms, rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rPROSACWITHNORM:
        status = a_prosac_nd(ms, rm, cm.oMatchedPts, td.oMatchedPts, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rVBAYESONLY:
    case sacType::rVBAYESWITHKNN:
        std::cout << "start-";
        status = vbayes_dcp(ms, cm, td, selectedcm, selectedtd);
        break;
    }
    return status;
}

bool sac::sactp_s(const methods& ms, const ransacMode rm, std::vector<cv::Point2d>& cmall, std::vector<cv::Point2d>& tdall, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, double th, bool clflag)
{
    bool status = true;
    switch (ms.rt) {
    case sacType::rRANSAC:
        status = a_ransac(ms, rm, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rRANSACWITHNORM:
        status = a_ransac_nd(ms, rm, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rREINFORCEMENT:
        status = a_ransac_rfl(ms, rm, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rREINFORCEMENTWITHNORM:
        status = a_ransac_rfl_nd(ms, rm, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rKERNELDE:
        status = a_kde(ms, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rKERNELDEWITHNORM:
        status = a_kde_nd(ms, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rPROSAC:
        status = a_prosac(ms, rm, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    case sacType::rPROSACWITHNORM:
        status = a_prosac_nd(ms, rm, cmall, tdall, selectedcm, selectedtd, th, clflag);
        break;
    }
    return status;
}
*/