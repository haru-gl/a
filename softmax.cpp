#include <math.h>
#include <iostream>
#include <vector>
#include <random>
#include "tmatrix.h"

void disprawprop(const std::vector<double>& prop)
{
    cv::Mat graf = cv::Mat::zeros(100, 1000, CV_8UC3);
    double dx = 1000.0 / (double)prop.size(), dy = 100.0;
    double max = -100000.0, min = 1000000.0;;
    double ave = 0.0, dev = 0.0;
    for (size_t i = 0; i < prop.size(); i++) {
        double ep= prop[i];
        ave += ep; dev += ep * ep;
        if (ep > max) max = ep;
        if (ep < min) min = ep;
    }
    ave /= (double)prop.size(); dev = dev / (double)prop.size() - ave * ave;
    dy = 100.0 / (max - min);
    for (size_t i = 0; i < prop.size(); i++) {
        int x = (int)(dx * (double)i);
        int y = (int)(100.0 - dy * (prop[i] - min));
        cv::line(graf, cv::Point(x, 100), cv::Point(x, y), cv::Scalar(255, 0, 255));
    }
    cv::imshow("RAW Prop", graf);
    int ky = cv::waitKey(1);
    std::cout << "RAW Prop: max=" << max << " :min=" << min << " :ave = " << ave << " :dev = " << dev << std::endl;
}

void dispexpprop(const std::vector<double>& prop)
{
    cv::Mat graf = cv::Mat::zeros(100, 1000, CV_8UC3);
    double dx = 1000.0 / (double)prop.size(), dy = 100.0;
    double max = -100000.0, min = 1000000.0;
    double sum = 0.0;// , dev, ave;
    for (size_t i = 0; i < prop.size(); i++) {
        double ep = exp(prop[i]);
        sum += ep; //dev += ep * ep;
        if (ep > max) max = ep;
        if (ep < min) min = ep;
    }
    max /= sum; min /= sum;
    dy = 100.0 / (max - min);
    for (size_t i = 0; i < prop.size(); i++) {
        int x = (int)(dx * (double)i);
        int y = (int)(100.0 - dy * (exp(prop[i]) / sum - min));
        cv::line(graf, cv::Point(x, 100), cv::Point(x, y), cv::Scalar(255, 0, 0));
    }
    cv::imshow("Exp Prop", graf);
    int ky = cv::waitKey(1);
    std::cout << "Exp Prop: max=" << max << " :min=" << min << std::endl;
}

void dispsqrtprop(const std::vector<double>& prop)
{
    cv::Mat graf = cv::Mat::zeros(100, 1000, CV_8UC3);
    double dx = 1000.0 / (double)prop.size(), dy = 100.0;
    double max = -100000.0, min = 1000000.0;
    double sum = 0.0;// , dev, ave;
    for (size_t i = 0; i < prop.size(); i++) {
        double ep = 1000.0 * sqrt(prop[i]) + 1.0;
        sum += ep; //dev += ep * ep;
        if (ep > max) max = ep;
        if (ep < min) min = ep;
    }
    max /= sum; min /= sum;
    dy = 100.0 / (max - min);
    for (size_t i = 0; i < prop.size(); i++) {
        int x = (int)(dx * (double)i);
        int y = (int)(100.0 - dy * ((1000.0 * sqrt(prop[i]) + 1.0) / sum - min));
        cv::line(graf, cv::Point(x, 100), cv::Point(x, y), cv::Scalar(0, 0, 255));
    }
    cv::imshow("Sqrt Prop", graf);
    int ky = cv::waitKey(1);
    std::cout << "Sqrt Prop: max=" << max << " :min=" << min << std::endl;
}

void dispexpprop(const std::vector<double>& expprp, double th, const std::vector<size_t>& pv)
{
    cv::Mat graf = cv::Mat::zeros(100, 1000, CV_8UC3);
    double dx = 1000.0 / (double)expprp.size(), dy;
    double max = expprp[0], min = expprp[0];
    for (size_t i = 1; i < expprp.size(); i++) {
        if (expprp[i] > max) max = expprp[i];
        else if (expprp[i] < min) min = expprp[i];
    }
    dy = 100.0 / (max - min);
    for (size_t i = 0; i < expprp.size(); i++) {
        int x = (int)(dx * (double)i);
        int y = (int)(100.0 - dy * (expprp[i] - min));
        cv::line(graf, cv::Point(x, 100), cv::Point(x, y), cv::Scalar(0, 255, 255));
    }
    int y = (int)(100.0 - dy * (th - min));
    cv::line(graf, cv::Point(0, y), cv::Point(999, y), cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < pv.size(); i++) {
        int x = (int)(dx * (double)pv[i]);
        int y = (int)(100.0 - dy * (expprp[pv[i]] - min));
        int yl = (int)(100.0 - dy * (th - min));
        cv::line(graf, cv::Point(x, yl), cv::Point(x, y), cv::Scalar(0, 0, 255));
    }
    cv::imshow("ExpProp", graf);
    int ky = cv::waitKey(1);
}
void disprawprop(const std::vector<double>& prop, double th, const std::vector<size_t>& pv, cv::Scalar cl)
{
    cv::Mat graf = cv::Mat::zeros(100, 1000, CV_8UC3);
    double dx = 1000.0 / (double)prop.size(), dy;
    double max = prop[0], min = prop[0];
    for (size_t i = 1; i < prop.size(); i++) {
        if (prop[i] > max) max = prop[i];
        else if (prop[i] < min) min = prop[i];
    }
    dy = 100.0 / (max - min);
    for (size_t i = 0; i < prop.size(); i++) {
        int x = (int)(dx * (double)i);
        int y = (int)(100.0 - dy * (prop[i] - min));
        cv::line(graf, cv::Point(x, 100), cv::Point(x, y), cv::Scalar(0, 255, 255));
    }
    int y = (int)(100.0 - dy * (th - min));
    cv::line(graf, cv::Point(0, y), cv::Point(999, y), cv::Scalar(255, 255, 255));
    for (size_t i = 0; i < pv.size(); i++) {
        int x = (int)(dx * (double)pv[i]);
        int y = (int)(100.0 - dy * (prop[pv[i]] - min));
        int yl = (int)(100.0 - dy * (th - min));
        cv::line(graf, cv::Point(x, yl), cv::Point(x, y), cl);
    }
    cv::imshow("RAW Prop", graf);
    int ky = cv::waitKey(1);
}

std::vector<size_t> softmax_y(size_t numPts, const std::vector<double>& prop, size_t sampleSize)
{
    std::vector<size_t> pv(sampleSize);
    int n = 0;
    double softunder = 0.0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> randp(0.0, 1.0);
    //dispexpprop(prop);
    for (int i = 0; i < numPts; i++) {
        double softone = 1000.0 * sqrt(prop[i]) + 1.0;
        softunder += softone;
    }
    while (n < sampleSize) {
        double p = randp(mt);
        double soft = 0.0;
        for (size_t i = 0; i < numPts; i++) {
            double softone = 1000.0 * sqrt(prop[i]) + 1.0;
            soft += softone / softunder;
            if (soft >= p) {
                pv[n] = i;
                int j;
                for (j = 0; j < n; j++)
                    if (pv[j] == pv[n]) break;
                if (j == n) n++;
                break;
            }
        }
    }
    return pv;
}

std::vector<size_t> softmax_y1(size_t numPts, const std::vector<double>& prop, size_t sampleSize)
{
    std::vector<size_t> pv(sampleSize);
    int n = 0;
    double softunder = 0.0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> randp(0.0, 1.0);
    //dispexpprop(prop);
    for (int i = 0; i < numPts; i++) {
        double softone = exp(prop[i]);
        softunder += softone;
    }
    while (n < sampleSize) {
        double p = randp(mt);
        double soft = 0.0;
        for (size_t i = 0; i < numPts; i++) {
            double softone = exp(prop[i]);
            soft += softone / softunder;
            if (soft >= p) {
                pv[n] = i;
                int j;
                for (j = 0; j < n; j++)
                    if (pv[j] == pv[n]) break;
                if (j == n) n++;
                break;
            }
        }
    }
    return pv;
}
std::vector<size_t> softmax_y2(size_t numPts, const std::vector<double>& prop, size_t sampleSize)
{
    std::vector<size_t> pv(sampleSize);
    int n = 0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //    std::uniform_real_distribution<> randp(0.0, 1.0);
    //dispexpprop(prop);

    std::vector<double> eprp(numPts);
    double softdenominator = 0.0;
    for (size_t i = 0; i < numPts; i++) {
        eprp[i] = exp(prop[i]);
        softdenominator += eprp[i];
    }
    eprp[0] /= softdenominator;
    double epmax = eprp[0], epmin = eprp[0];
    double ave = 0.0, dev = 0.0;
    for (size_t i = 1; i < numPts; i++) {
        eprp[i] /= softdenominator;
        if (epmax < eprp[i]) epmax = eprp[i];
        else if (epmin > eprp[i]) epmin = eprp[i];
        ave += eprp[i]; dev += eprp[i] * eprp[i];
    }
    ave /= (double)numPts; dev = dev / (double)numPts - ave * ave;
    double epwd = (epmax - epmin) / 10.0;
    printf("epmin=%g,epmax=%g,ave=%g,dev=%g:", epmin, epmax, ave, dev);
    if (epwd < 1.0e-5 || dev > 1.0e-8) {
        pv = randperm(numPts, sampleSize);
        printf("wd=%f,rand\n", epwd);
        return pv;
    }
    std::uniform_real_distribution<> randp(epmin + epwd, epmax - epwd);
    //この乱数の出し方
    //下限を設けることで、確率的に有効ではない対応点が選択されることを避ける。現状は最小値の1割増しだが、変更は可能。
    //上限を設けることで、選ばれる対応点数を確保する。現状は最大値の1割引きだが、変更は可能。

    std::vector<size_t> pvc; pvc.clear();
    double th;
    while (1) {
        th = randp(mt);//確率p以上の対応点を探す。
        //printf("p=%f\n", p);
        for (size_t i = 0; i < numPts; i++)
            if (eprp[i] >= th) pvc.push_back(i);//見つかった対応点の位置iをためる
        if (pvc.size() >= sampleSize) break;//見つかった対応点数がsampleSize個以上なら抜ける。
        pvc.clear();//配列を空にして探しなおし。
    }
    printf("th=%f,Num=%d, Reinforcement\n", th, (int)pvc.size());
    dispexpprop(eprp, th, pvc);

    std::vector<size_t> pvn(sampleSize);
    pvn = randperm(pvc.size(), sampleSize);//見つかった対応点数の中からランダムにsampleSize個の場所を選び配列pvnに並べる
    for (size_t i = 0; i < sampleSize; i++)//選ばれた場所pnv[i]のpvc（対応点の場所）を、配列pvにsampleSize個並べる
        pv[i] = pvc[pvn[i]];
    pvc.clear(); pvc.shrink_to_fit();
    pvn.clear(); pvn.shrink_to_fit();
    return pv;
}
std::vector<size_t> softmax_y3(size_t numPts, const std::vector<double>& prop, size_t sampleSize)
{
    std::vector<size_t> pv(sampleSize);
    int n = 0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //    std::uniform_real_distribution<> randp(0.0, 1.0);
    //dispexpprop(prop);
    std::vector<double> eprp(numPts);
    double softdenominator = 0.0;
    for (size_t i = 0; i < numPts; i++) {
        eprp[i] = exp(prop[i]);
        softdenominator += eprp[i];
    }
    eprp[0] /= softdenominator;
    double epmax = eprp[0], epmin = eprp[0];
    double ave = 0.0, dev = 0.0;
    for (size_t i = 1; i < numPts; i++) {
        eprp[i] /= softdenominator;
        if (epmax < eprp[i]) epmax = eprp[i];
        else if (epmin > eprp[i]) epmin = eprp[i];
        ave += eprp[i]; dev += eprp[i] * eprp[i];
    }
    ave /= (double)numPts; dev = dev / (double)numPts - ave * ave;
    double epwd = (epmax - epmin) / 10.0;
    printf("epmin=%g,epmax=%g,ave=%g,dev=%g:", epmin, epmax, ave, dev);
    if (epwd < 1.0e-8) {
        pv = randperm(numPts, sampleSize);
        printf("wd=%f,rand0\n", epwd);
        return pv;
    }
    if (dev > 1.0e-8) {
        pv = randperm(numPts, sampleSize);
        printf("dev=%f,rand1\n", dev);
        return pv;
    }
    //    std::uniform_real_distribution<> randp(epmin + epwd, epmax - epwd);
    //この乱数の出し方
    //下限を設けることで、確率的に有効ではない対応点が選択されることを避ける。現状は最小値の1割増しだが、変更は可能。
    //上限を設けることで、選ばれる対応点数を確保する。現状は最大値の1割引きだが、変更は可能。

    std::vector<size_t> pvc;
    double th = (epmax + epmin) / 2.0, thw = (epmax - epmin) / 20.0;//閾値の初期値を中間に設定
    int it;
    for (it = 0; it < 30; it++) {//th-=thwとth+=thwを繰り返すことがあるので、有限回の繰り返しにする
        pvc.clear();//配列を空にして探しなおし。
        printf("th=%f:", th);
        if (th < epmin || th > epmax) {
            pv = randperm(numPts, sampleSize);
            printf("not found, rand2\n");
            return pv;
        }
        for (size_t i = 0; i < numPts; i++)
            if (eprp[i] >= th) pvc.push_back(i);//見つかった対応点の位置iをためる
        if (pvc.size() < sampleSize) {
            printf("th down"); th -= thw;
        }//見つかった対応点がsampleSize個未満なら閾値thを下げる
        else if (pvc.size() < sampleSize * 20) {
            printf("th up"); th -= thw * 0.25;
        }
        else if (pvc.size() > sampleSize * 100) {
            printf("th more up"); th += thw * 0.5;
        }//見つかった対応点が多すぎたら、閾値thを上げる
        else break;
    }
    if (it == 30 && pvc.size() < sampleSize) {//thの調整が多すぎ、かつ所定の対応点数を見つけられなかったら乱数
        pv = randperm(numPts, sampleSize);
        printf("th search, max iteration, rand3\n");
        return pv;
    }
    printf("th=%f,Num=%d, Reinforcement\n", th, (int)pvc.size());
    dispexpprop(eprp, th, pvc);

    std::vector<size_t> pvn(sampleSize);
    pvn = randperm(pvc.size(), sampleSize);//見つかった対応点数の中からランダムにsampleSize個の場所を選び配列pvnに並べる
    for (size_t i = 0; i < sampleSize; i++)//選ばれた場所pnv[i]のpvc（対応点の場所）を、配列pvにsampleSize個並べる
        pv[i] = pvc[pvn[i]];
    pvc.clear(); pvc.shrink_to_fit();
    pvn.clear(); pvn.shrink_to_fit();
    return pv;
}

std::vector<size_t> softmax_n(size_t numPts, const std::vector<double>& prop, size_t sampleSize)
{
    std::vector<size_t> pv(sampleSize);
    int n = 0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //    std::uniform_real_distribution<> randp(0.0, 1.0);

//  prop[]内のzeroの個数をカウント。50％以上でRANSAC
    size_t znum = 0;
    for (size_t i = 0; i < prop.size(); i++)
        if (prop[i] < 1.0e-9) znum++;
    if ((double)znum / (double)prop.size() > 0.5) {
        pv = randperm(numPts, sampleSize);
        std::cout << "too match zero count=" << znum << "/" << prop.size() << " :rand0" << std::endl;
        return pv;
    }

//  prop[]の分散計算
    double epmax = prop[0], epmin = prop[0];
    double ave = 0.0, dev = 0.0;
    for (size_t i = 1; i < numPts; i++) {
        double prp = prop[i];
        if (epmax < prp) epmax = prp;
        else if (epmin > prp) epmin = prp;
        ave += prp; dev += prp * prp;
    }
    ave /= (double)numPts; dev = dev / (double)numPts - ave * ave;
    double epwd = (epmax - epmin) / 10.0;
    //std::cout << "min=" << epmin << " max=" << epmax << " ave=" << ave << " dev=" << dev << std::endl;
    if (epwd < 1.0e-8) {
        pv = randperm(numPts, sampleSize);
        std::cout << "narrow width(" << epwd << "):rand1" << std::endl;
        return pv;
    }
    if (dev > 1.0e-1) {
        pv = randperm(numPts, sampleSize);
        std::cout << " high deviation(" << dev << "):rand2" << std::endl;
        return pv;
    }
    //    std::uniform_real_distribution<> randp(epmin + epwd, epmax - epwd);
    std::vector<size_t> pvc;
    double th = (epmax + epmin) / 2.0, thw = (epmax - epmin) / 20.0;//閾値の初期値を中間に設定
    int it;
    for (it = 0; it < 30; it++) {//th-=thwとth+=thwを繰り返すことがあるので、有限回の繰り返しにする
        pvc.clear();//配列を空にして探しなおし。
        //std::cout << " th=" << th;
        if (th < epmin || th > epmax) {
            pv = randperm(numPts, sampleSize);
            std::cout << "th learning false: rand3" << std::endl;
            return pv;
        }

        for (size_t i = 0; i < numPts; i++)
            if (prop[i] >= th) pvc.push_back(i);//見つかった対応点の位置iをためる

        if (pvc.size() < sampleSize) {//見つかった対応点がsampleSize個未満なら閾値thを下げる
            /*printf("th down");*/ th -= thw;
        }
        else if (pvc.size() < sampleSize * 20) {//見つかった対応点がsampleSize個*20未満なら閾値thを少し下げる
            /*printf("th up");*/ th -= thw * 0.25;
        }
        else if (pvc.size() > sampleSize * 100) {//見つかった対応点が多すぎたら、閾値thを上げる
            /*printf("th more up");*/ th += thw * 0.5;
        }
        else break;
    }
    if (pvc.size() < sampleSize) {//thの調整が多すぎ、かつ所定の対応点数を見つけられなかったら乱数
        pv = randperm(numPts, sampleSize);
        std::cout << "max iteration, less sampleSize: rand4" << std::endl;
        return pv;
    }
    std::cout << "th=" << th << " Num=" << pvc.size() << " :Reinforcement" << std::endl;
    disprawprop(prop, th, pvc, cv::Scalar(0,0,255));

    std::vector<size_t> pvn(sampleSize);
    pvn = randperm(pvc.size(), sampleSize);//見つかった対応点数の中からランダムにsampleSize個の場所を選び配列pvnに並べる
    for (size_t i = 0; i < sampleSize; i++)//選ばれた場所pnv[i]のpvc（対応点の場所）を、配列pvにsampleSize個並べる
        pv[i] = pvc[pvn[i]];
    pvc.clear(); pvc.shrink_to_fit();
    pvn.clear(); pvn.shrink_to_fit();
    return pv;
}

//status
//=0: Normaly Reinforcement
//=1: Many Prop[]'s are zero's; 
//=2: Width of value of Prop[] is to narrow;
//=3: Deviation of Prop[] is to large;
//=4: Th level is to low;
//=5: Th level is to high;
//=6: Less sampleSize;
std::vector<size_t> softmax_n1(size_t numPts, const std::vector<double>& prop, size_t sampleSize, int& status)
{
    cv::Scalar cl = cv::Scalar(0, 0, 255);
    status = 0;
    std::vector<size_t> pv(sampleSize);
    int n = 0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    //    std::uniform_real_distribution<> randp(0.0, 1.0);

//  prop[]内のzeroの個数をカウント。50％以上でRANSAC
    size_t znum = 0;
    for (size_t i = 0; i < prop.size(); i++)
        if (prop[i] < 1.0e-9) znum++;
    if ((double)znum / (double)prop.size() > 0.5) {
        pv = randperm(numPts, sampleSize);
        std::cout << "too match zero count=" << znum << "/" << prop.size() << " :rand0" << std::endl;
        status = 1;
        return pv;
    }

    //  prop[]の分散計算
    double epmax = prop[0], epmin = prop[0];
    double ave = 0.0, dev = 0.0;
    for (size_t i = 1; i < numPts; i++) {
        double prp = prop[i];
        if (epmax < prp) epmax = prp;
        else if (epmin > prp) epmin = prp;
        ave += prp; dev += prp * prp;
    }
    ave /= (double)numPts; dev = dev / (double)numPts - ave * ave;
    double epwd = (epmax - epmin) / 10.0;
    //std::cout << "min=" << epmin << " max=" << epmax << " ave=" << ave << " dev=" << dev << std::endl;
    if (epwd < 1.0e-8) {
        pv = randperm(numPts, sampleSize);
        std::cout << "narrow width(" << epwd << "):rand1" << std::endl;
        status = 2;
        return pv;
    }
    if (dev > 1.0e-1) {
        pv = randperm(numPts, sampleSize);
        std::cout << " high deviation(" << dev << "):rand2" << std::endl;
        status = 3;
        return pv;
    }
    //    std::uniform_real_distribution<> randp(epmin + epwd, epmax - epwd);
    std::vector<size_t> pvc;
    double th = (epmax + epmin) / 4.0, thw = (epmax - epmin) / 20.0;//閾値の初期値を中間に設定
    int it;
    for (it = 0; it < 30; it++) {//th-=thwとth+=thwを繰り返すことがあるので、有限回の繰り返しにする
        pvc.clear();//配列を空にして探しなおし。
        //std::cout << " th=" << th;
        if (th < epmin) {
            th = epmin;
//            pv = randperm(numPts, sampleSize);
//            std::cout << "th is to low: rand3" << std::endl;
              status = 4;
//            return pv;
        }
        if (th > epmax) {
            th = epmax;
//            pv = randperm(numPts, sampleSize);
//            std::cout << "th is to high: rand4" << std::endl;
              status = 5;
//            return pv;
        }

        for (size_t i = 0; i < numPts; i++)
            if (prop[i] >= th) pvc.push_back(i);//見つかった対応点の位置iをためる

        if (pvc.size() < sampleSize) {//見つかった対応点がsampleSize個未満なら閾値thを下げる
            /*printf("th down");*/ th -= thw;
        }
        else if (pvc.size() < sampleSize * 30) {//見つかった対応点がsampleSize個*20未満なら閾値thを少し下げる
            /*printf("th up");*/ th -= thw * 0.25;
        }
        else if (pvc.size() > sampleSize * 125) {//見つかった対応点が多すぎたら、閾値thを上げる
            /*printf("th more up");*/ th += thw * 0.5;
        }
        else break;
    }
    if (pvc.size() < sampleSize) {//thの調整が多すぎ、かつ所定の対応点数を見つけられなかったら乱数
        pv = randperm(numPts, sampleSize);
        std::cout << "max iteration, less sampleSize: rand5" << std::endl;
        status = 6;
        return pv;
    }
    std::cout << "th=" << th << " Num=" << pvc.size() << " :Reinforcement" << std::endl;
    disprawprop(prop, th, pvc, cl);

    std::vector<size_t> pvn(sampleSize);
    pvn = randperm(pvc.size(), sampleSize);//見つかった対応点数の中からランダムにsampleSize個の場所を選び配列pvnに並べる
    for (size_t i = 0; i < sampleSize; i++)//選ばれた場所pnv[i]のpvc（対応点の場所）を、配列pvにsampleSize個並べる
        pv[i] = pvc[pvn[i]];
    pvc.clear(); pvc.shrink_to_fit();
    pvn.clear(); pvn.shrink_to_fit();
    //status = 0;
    return pv;
}



std::vector<size_t> softmax(size_t numPts, const std::vector<double>& prop, size_t sampleSize)
{
    std::vector<size_t> pv(sampleSize);
    bool dup = false;
    int n = 0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> randp(0.80, 1.0), randq(0.0, 1.0);
    int zcnt = 0;
    //dispexpprop(prop);
    for (size_t i = 0; i < numPts; i++)
        if (fabs(prop[i]) < 1.0e-8) zcnt++;
    //printf("zero count=%d:", zcnt);
    if ((double)zcnt / (double)numPts > randp(mt)) {
        printf("-");
        pv = randperm(numPts, sampleSize);
    }
    else {
        double softdenominator = 0.0;
        for (size_t i = 0; i < numPts; i++) {
            double softone = exp(prop[i]);
            softdenominator += softone;
        }
        std::vector<size_t> mx;
        mx.push_back(0);
        double maxprop = exp(prop[0]) / softdenominator;
        for (size_t i = 1; i < numPts; i++) {
            double np = exp(prop[i]) / softdenominator;
            if (np >= maxprop) {
                maxprop = np;
                mx.push_back(i);
            }
        }
        if (mx.size() >= sampleSize) {
            printf("*");
            for (size_t i = 0; i < sampleSize; i++)
                pv[i] = mx[mx.size() - 1 - i];
        }
        else {
            printf("/");
            pv = randperm(numPts, sampleSize);
        }
        mx.clear(); mx.shrink_to_fit();
    }
    return pv;
}

std::vector<size_t> softmax_select(size_t numPts, const std::vector<double>& prop, size_t& sampleNum)
{
    std::vector<size_t> pv;
    double softdenominator = 0.0;
    for (size_t i = 0; i < numPts; i++) {
        double softone = exp(prop[i]);
        softdenominator += softone;
    }
    std::vector<size_t> mx;
    double maxprop = 1.0 / softdenominator;
    for (size_t i = 0; i < numPts; i++) {
        double np = exp(prop[i]) / softdenominator;
        if (np >= maxprop) {
            maxprop = np;
            mx.push_back(i);
        }
    }
    if (mx.size() == 0) {
        printf("/");
        pv = randperm(numPts, sampleNum);
    }
    else if (mx.size() >= sampleNum) {
        printf("*");
        for (size_t i = 0; i < sampleNum; i++)
            pv.push_back(mx[mx.size() - 1 - i]);
    }
    else {
        printf("+");
        sampleNum = mx.size();
        for (size_t i = 0; i < sampleNum; i++)
            pv.push_back(mx[mx.size() - 1 - i]);
    }
    mx.clear(); mx.shrink_to_fit();
    return pv;
}

std::vector<size_t> softmax_org(size_t numPts, const std::vector<double>& prop, size_t& sampleNum, size_t cnt)
{
    std::vector<size_t> pv(sampleNum);
    bool dup = false;
    int n = 0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> randp(0.0, 1.0), randq(0.0, 1.0);

    //dispexpprop(prop);
    if (cnt < 100) {
        printf("-");
        pv = randperm(numPts, sampleNum);
    }
    else if (cnt < 1000) {
        int pp = (cnt / 100) * 10;
        if (cnt % pp == 0) {
            printf("-");
            pv = randperm(numPts, sampleNum);
        }
        else pv = softmax_select(numPts, prop, sampleNum);
    }
    else pv = softmax_select(numPts, prop, sampleNum);

    return pv;
}

std::vector<size_t> softmax_dc(size_t numPts, const std::vector<double>& disvalue, size_t sampleSize) //直接誤差を使ってsoftmax
{
    std::vector<size_t> pv(sampleSize);
    int n = 0;
    double softunder = 0.0;
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<> randp(0.0, 1.0);
    //dispexpprop(prop);
    for (int i = 0; i < numPts; i++) {
        double softone = 1000.0 * sqrt(disvalue[i]) + 1.0;
        softunder += softone;
    }
    while (n < sampleSize) {
        double p = randp(mt);
        double soft = 0.0;
        for (size_t i = 0; i < numPts; i++) {
            double softone = 1000.0 * sqrt(disvalue[i]) + 1.0;
            soft += softone / softunder;
            if (soft >= p) {
                pv[n] = i;
                int j;
                for (j = 0; j < n; j++)
                    if (pv[j] == pv[n]) break;
                if (j == n) n++;
                break;
            }
        }
    }
    return pv;
}
