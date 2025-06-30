#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "classes.h"
#include "sac.h"
#include "prosac.h"
#include "tmatrix.h"

bool vbayes_dcp(const methods& ms, clipedmap_data& clpd, target_data& tgt, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd);

void sac::set_maxIteration(int th)
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

void get_ave_stddev(const std::vector<double>& exy, double &nave, double &nstddv)
{
    double ave = 0.0,dev = 0.0;
    for(size_t j = 0; j < exy.size(); j++){
        ave += exy[j];
        dev += exy[j] * exy[j];
    }
    ave /= (double)exy.size();
    dev /= (double)exy.size();
    dev -= ave * ave;
    nave = ave; nstddv = sqrt(dev);
}

double get_median(std::vector<double>& exy)
{
    double med;
    std::sort(exy.begin(), exy.end());
    size_t ecentor = exy.size() / 2;
    if(exy.size() % 2 == 1 )
        med = exy[ecentor];
    else
        med= (exy[ecentor] + exy[ecentor + 1]) / 2.0;
    return med;
}
// basic ransac
bool sac::a_ransac(const methods& ms, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(ms);
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
 
    switch(rm){
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
        cv::Mat tform = computematrix(ms, samplePts1, samplePts2);

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
    for(size_t j = 0; j < exy.size(); j++)
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
bool sac::a_ransac_nd(const methods& ms, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(ms);
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

    switch(rm){
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
        cv::Mat tform = computematrix(ms, normalizedPts1, normalizedPts2);
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
    for(size_t j = 0; j < exy.size(); j++)
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

// ransac with reinforcemant learning
bool sac::a_ransac_rfl(const methods& ms, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(ms);
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

    switch(rm){
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
        indices = softmax(numPts, prop);

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(ms, samplePts1, samplePts2);

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
                disvalue[j] = (tsd - norm) / tsd;
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
            if (indices[0] != j && indices[1] != j && indices[2] != j && indices[3] != j) {
                prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
            }
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
    for(size_t j = 0; j < exy.size(); j++)
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

// ransac with reinforcemant learning with normalization, denormalization(–¢Š®¬)
bool sac::a_ransac_rfl_nd(const methods& ms, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(ms);
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

    switch(rm){
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
		indices = softmax(numPts, prop);

		for (int i = 0; i < sampleSize; i++) {
			samplePts1[i] = td[indices[i]];
			samplePts2[i] = cm[indices[i]];
		}

		normalization(samplePts1, normalizedPts1, normalizedMat1);
		normalization(samplePts2, normalizedPts2, normalizedMat2);
		cv::Mat tform = computematrix(ms, normalizedPts1, normalizedPts2);
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
				disvalue[j] = (tsd - norm) / tsd;
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
			if (indices[0] != j && indices[1] != j && indices[2] != j && indices[3] != j) {
				prop[j] += sumvalue / maxvalue * alpha * (disvalue[j] - prop[j]);
			}
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
    for(size_t j = 0; j < exy.size(); j++)
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

//ransac by kernel density estimation
bool sac::a_kde(const methods& ms,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, 
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(ms);
    size_t numPts = td.size();
    std::vector<size_t> indices = randperm(numPts, sampleSize);
    std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
    cv::Mat tform;
    cv::Point2d pcenter(SZCENTER, SZCENTER);

    for (size_t it = 1; it <= kdeIteration; it++) {
        std::cout << it << ":";
        indices = randperm(numPts, sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }

        tform = computematrix(ms, samplePts1, samplePts2);
        cv::Point2d estimated = transform2d(pcenter, tform);
        addKernel2map(estimated);
        //        cv::Mat img; heatmap(img);
        //        cv::imshow("Heatmap", img);
        //        cv::waitKey(1);
    }
    kdmap /= (double)kdeIteration;
    cv::Point2d detectedpeak = peakdetection();

    size_t bestInliersNum = setnum * sampleSize;
    std::vector<size_t> goodInliersIdx(bestInliersNum);
    int snum = 0;
    disterr = th;
    while (1) {
        indices = randperm(numPts, sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        tform = computematrix(ms, samplePts1, samplePts2);
        cv::Point2d estimated = transform2d(pcenter, tform);
        double dist = cv::norm(estimated - detectedpeak);
        if (dist < disterr) {
            for (int i = 0; i < sampleSize; i++)
                goodInliersIdx[snum * sampleSize + i] = indices[i];
            snum++;    if (snum >= setnum) break;
        }
    }
    if (bestInliersNum < sampleSize) return true;

    if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
    for (size_t i = 0; i < bestInliersNum; i++) {
        selectedtd.push_back(td[goodInliersIdx[i]]);
        selectedcm.push_back(cm[goodInliersIdx[i]]);
    }

    goodInliersIdx.clear(); goodInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();

    return false;
}

//ransac by kernel density estimation with normalization, denormalization(–¢Š®¬)
bool sac::a_kde_nd(const methods& ms,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td, 
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, 
    double th, bool clflag)
{
	const size_t sampleSize = (size_t)get_minGP(ms);
	size_t numPts = td.size();
	std::vector<size_t> indices = randperm(numPts, sampleSize);
	std::vector<cv::Point2d> samplePts1(sampleSize), samplePts2(sampleSize);
	cv::Mat tform;
	cv::Point2d pcenter(SZCENTER, SZCENTER);
	std::vector<cv::Point2d> normalizedPts1(sampleSize);
	cv::Mat normalizedMat1;
	std::vector<cv::Point2d> normalizedPts2(sampleSize);
	cv::Mat normalizedMat2;

	for (size_t it = 1; it <= kdeIteration; it++) {
		std::cout << it << ":";
		indices = randperm(numPts, sampleSize);
		for (int i = 0; i < sampleSize; i++) {
			samplePts1[i] = td[indices[i]];
			samplePts2[i] = cm[indices[i]];
		}

		normalization(samplePts1, normalizedPts1, normalizedMat1);
		normalization(samplePts2, normalizedPts2, normalizedMat2);
		cv::Mat tform = computematrix(ms, normalizedPts1, normalizedPts2);
		denormalization(tform, normalizedMat1, normalizedMat2);

		cv::Point2d estimated = transform2d(pcenter, tform);
		addKernel2map(estimated);
		//		cv::Mat img; heatmap(img);
		//		cv::imshow("Heatmap", img);
		//		cv::waitKey(1);
	}
	kdmap /= (double)kdeIteration;
	cv::Point2d detectedpeak = peakdetection();

	size_t bestInliersNum = setnum * sampleSize;
	std::vector<size_t> goodInliersIdx(bestInliersNum);
	int snum = 0;
    disterr = th;
	while (1) {
		indices = randperm(numPts, sampleSize);
		for (int i = 0; i < sampleSize; i++) {
			samplePts1[i] = td[indices[i]];
			samplePts2[i] = cm[indices[i]];
		}
		tform = computematrix(ms, samplePts1, samplePts2);
		cv::Point2d estimated = transform2d(pcenter, tform);
		double dist = cv::norm(estimated - detectedpeak);
		if (dist < disterr) {
			for (int i = 0; i < sampleSize; i++)
				goodInliersIdx[snum * sampleSize + i] = indices[i];
			snum++;	if (snum >= setnum) break;
		}
	}
	if (bestInliersNum < sampleSize) return true;

	if (clflag == true) { selectedtd.clear(); selectedcm.clear(); }
	for (size_t i = 0; i < bestInliersNum; i++) {
		selectedtd.push_back(td[goodInliersIdx[i]]);
		selectedcm.push_back(cm[goodInliersIdx[i]]);
	}

    goodInliersIdx.clear(); goodInliersIdx.shrink_to_fit();
    indices.clear(); indices.shrink_to_fit();
    samplePts1.clear(); samplePts1.shrink_to_fit();
    samplePts2.clear(); samplePts2.shrink_to_fit();
    normalizedPts1.clear(); normalizedPts1.shrink_to_fit();
    normalizedPts2.clear(); normalizedPts2.shrink_to_fit();

	return false;
}

// prosac
bool sac::a_prosac(const methods& ms, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(ms);
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
    prosac randperm; randperm.init_prosac(numPts, sampleSize);
    
    switch(rm){
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
        indices = randperm.prosac_sampling();

        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }
        cv::Mat tform = computematrix(ms, samplePts1, samplePts2);

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
    for(size_t j = 0; j < exy.size(); j++)
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

// prosac with normalization and denormalization
bool sac::a_prosac_nd(const methods& ms, const ransacMode rm,
    std::vector<cv::Point2d>& cm, std::vector<cv::Point2d>& td,
    std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd,
    double th, bool clflag)
{
    const size_t sampleSize = (size_t)get_minGP(ms);
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
    prosac randperm; randperm.init_prosac(numPts, sampleSize);

    switch(rm){
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
        indices = randperm.prosac_sampling();
        
        for (int i = 0; i < sampleSize; i++) {
            samplePts1[i] = td[indices[i]];
            samplePts2[i] = cm[indices[i]];
        }

        normalization(samplePts1, normalizedPts1, normalizedMat1);
        normalization(samplePts2, normalizedPts2, normalizedMat2);
        cv::Mat tform = computematrix(ms, normalizedPts1, normalizedPts2);
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
    for(size_t j = 0; j < exy.size(); j++)
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

bool sac::matrixestimation(const methods &ms, std::vector<cv::Point2d> &selectedcm, std::vector<cv::Point2d> &selectedtd, cv::Mat &tform)
{
	bool status = true;
	cv::Mat x0, xn1;
	switch (ms.ct) {
    case matrixcalType::cSVD:
		tform = computematrix(ms, selectedtd, selectedcm);
		break;
    case matrixcalType::cSVD_EIGEN:
		tform = computematrix_byEigen(ms, selectedtd, selectedcm);
		break;
    case matrixcalType::cGAUSSNEWTON:
		tform = computematrix(ms, selectedtd, selectedcm);
		status = checkFunc(tform);
		x0 = cnv_mt2vc(ms, tform);
		xn1 = gauss_newton(ms, selectedtd, selectedcm, x0);
		tform = cnv_vc2mt(ms, xn1);
		break;
    case matrixcalType::cGAUSSNEWTON_EIGEN:
		tform = computematrix_byEigen(ms, selectedtd, selectedcm);
		status = checkFunc(tform);
		x0 = cnv_mt2vc(ms, tform);
		xn1 = gauss_newton(ms, selectedtd, selectedcm, x0);
		tform = cnv_vc2mt(ms, xn1);
		break;
	}
	status = checkFunc(tform);
	return status;
}

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
