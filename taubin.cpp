#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


//Taubin法ヤコビ行列の成分と微分列
cv::Mat set_ks(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2, int a, int k)
{
    cv::Mat ks = cv::Mat::zeros(9, 1, CV_64FC1);
    switch (k) {
    case 0:
//        ks.at<double>(0, 0) = 0.0;                ks.at<double>(1, 0) = 0.0;                ks.at<double>(2, 0) = 0.0;
//        ks.at<double>(3, 0) = -p1[a].x;           ks.at<double>(4, 0) = -p1[a].y;           ks.at<double>(5, 0) = -1.0;
//        ks.at<double>(6, 0) = p1[a].x * p2[a].y;  ks.at<double>(7, 0) = p1[a].y * p2[a].y;  ks.at<double>(8, 0) = p2[a].y;
        ks.at<double>(3, 0) = -p1[a].x;           ks.at<double>(4, 0) = -p1[a].y;           ks.at<double>(5, 0) = -1.0;
        ks.at<double>(6, 0) = p1[a].x * p2[a].y;  ks.at<double>(7, 0) = p1[a].y * p2[a].y;  ks.at<double>(8, 0) = p2[a].y;
        break;
    case 1:
//        ks.at<double>(0, 0) = p1[a].x;            ks.at<double>(1, 0) = p1[a].y;            ks.at<double>(2, 0) = 1.0;
//        ks.at<double>(3, 0) = 0.0;                ks.at<double>(4, 0) = 0.0;                ks.at<double>(5, 0) = 0.0;
//        ks.at<double>(6, 0) = -p1[a].x * p2[a].x; ks.at<double>(7, 0) = -p1[a].y * p2[a].x; ks.at<double>(8, 0) = -p2[a].x;
        ks.at<double>(0, 0) = p1[a].x;            ks.at<double>(1, 0) = p1[a].y;            ks.at<double>(2, 0) = 1.0;
        ks.at<double>(6, 0) = -p1[a].x * p2[a].x; ks.at<double>(7, 0) = -p1[a].y * p2[a].x; ks.at<double>(8, 0) = -p2[a].x;
        break;
    case 2:
//        ks.at<double>(0, 0) = -p1[a].x * p2[a].y; ks.at<double>(1, 0) = -p1[a].y * p2[a].y; ks.at<double>(2, 0) = -p2[a].y;
//        ks.at<double>(3, 0) = p1[a].x * p2[a].x;  ks.at<double>(4, 0) = p1[a].y * p2[a].x;  ks.at<double>(5, 0) = p2[a].x;
//        ks.at<double>(6, 0) = 0.0;                ks.at<double>(7, 0) = 0.0;                ks.at<double>(8, 0) = 0.0;
        ks.at<double>(0, 0) = -p1[a].x * p2[a].y; ks.at<double>(1, 0) = -p1[a].y * p2[a].y; ks.at<double>(2, 0) = -p2[a].y;
        ks.at<double>(3, 0) = p1[a].x * p2[a].x;  ks.at<double>(4, 0) = p1[a].y * p2[a].x;  ks.at<double>(5, 0) = p2[a].x;
        break;
    }
    return ks;
}
cv::Mat set_T(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2, int a, int k)
{
    cv::Mat T = cv::Mat::zeros(9, 4, CV_64FC1);
    switch (k) {
    case 0:
//        T.at<double>(0, 0) = 0.0;      T.at<double>(0, 1) = 0.0;      T.at<double>(0, 2) = 0.0;      T.at<double>(0, 3) = 0.0;
//        T.at<double>(1, 0) = 0.0;      T.at<double>(1, 1) = 0.0;      T.at<double>(1, 2) = 0.0;      T.at<double>(1, 3) = 0.0;
//        T.at<double>(2, 0) = 0.0;      T.at<double>(2, 1) = 0.0;      T.at<double>(2, 2) = 0.0;      T.at<double>(2, 3) = 0.0;
//        T.at<double>(3, 0) = -1.0;     T.at<double>(3, 1) = 0.0;      T.at<double>(3, 2) = 0.0;      T.at<double>(3, 3) = 0.0;
//        T.at<double>(4, 0) = 0.0;      T.at<double>(4, 1) = -1.0;     T.at<double>(4, 2) = 0.0;      T.at<double>(4, 3) = 0.0;
//        T.at<double>(5, 0) = 0.0;      T.at<double>(5, 1) = 0.0;      T.at<double>(5, 2) = 0.0;      T.at<double>(5, 3) = 0.0;
//        T.at<double>(6, 0) = p2[a].y;  T.at<double>(6, 1) = 0.0;      T.at<double>(6, 2) = 0.0;      T.at<double>(6, 3) = p1[a].x;
//        T.at<double>(7, 0) = 0.0;      T.at<double>(7, 1) = p2[a].y;  T.at<double>(7, 2) = 0.0;      T.at<double>(7, 3) = p1[a].y;
//        T.at<double>(8, 0) = 0.0;      T.at<double>(8, 1) = 0.0;      T.at<double>(8, 2) = 0.0;      T.at<double>(8, 3) = 1.0;
        T.at<double>(3, 0) = -1.0;
        T.at<double>(4, 1) = -1.0;
        T.at<double>(6, 0) = p2[a].y;    
        T.at<double>(6, 3) = p1[a].x;
        T.at<double>(7, 1) = p2[a].y;
        T.at<double>(7, 3) = p1[a].y;
        T.at<double>(8, 3) = 1.0;
        break;
    case 1:
//        T.at<double>(0, 0) = 1.0;      T.at<double>(0, 1) = 0.0;      T.at<double>(0, 2) = 0.0;      T.at<double>(0, 3) = 0.0;
//        T.at<double>(1, 0) = 0.0;      T.at<double>(1, 1) = 1.0;      T.at<double>(1, 2) = 0.0;      T.at<double>(1, 3) = 0.0;
//        T.at<double>(2, 0) = 0.0;      T.at<double>(2, 1) = 0.0;      T.at<double>(2, 2) = 0.0;      T.at<double>(2, 3) = 0.0;
//        T.at<double>(3, 0) = 0.0;      T.at<double>(3, 1) = 0.0;      T.at<double>(3, 2) = 0.0;      T.at<double>(3, 3) = 0.0;
//        T.at<double>(4, 0) = 0.0;      T.at<double>(4, 1) = 0.0;      T.at<double>(4, 2) = 0.0;      T.at<double>(4, 3) = 0.0;
//        T.at<double>(5, 0) = 0.0;      T.at<double>(5, 1) = 0.0;      T.at<double>(5, 2) = 0.0;      T.at<double>(5, 3) = 0.0;
//        T.at<double>(6, 0) = -p2[a].x; T.at<double>(6, 1) = 0.0;      T.at<double>(6, 2) = -p1[a].x; T.at<double>(6, 3) = 0.0;
//        T.at<double>(7, 0) = 0.0;      T.at<double>(7, 1) = -p2[a].x; T.at<double>(7, 2) = -p1[a].y; T.at<double>(7, 3) = 0.0;
//        T.at<double>(8, 0) = 0.0;      T.at<double>(8, 1) = 0.0;      T.at<double>(8, 2) = -1.0;     T.at<double>(8, 3) = 0.0;
        T.at<double>(0, 0) = 1.0;
        T.at<double>(1, 1) = 1.0;
        T.at<double>(6, 0) = -p2[a].x;
        T.at<double>(6, 2) = -p1[a].x;
        T.at<double>(7, 1) = -p2[a].x; 
        T.at<double>(7, 2) = -p1[a].y;
        T.at<double>(8, 2) = -1.0;
        break;
    case 2:
//        T.at<double>(0, 0) = -p2[a].y; T.at<double>(0, 1) = 0.0;      T.at<double>(0, 2) = 0.0;      T.at<double>(0, 3) = -p1[a].x;
//        T.at<double>(1, 0) = 0.0;      T.at<double>(1, 1) = -p2[a].y; T.at<double>(1, 2) = 0.0;      T.at<double>(1, 3) = -p1[a].y;
//        T.at<double>(2, 0) = 0.0;      T.at<double>(2, 1) = 0.0;      T.at<double>(2, 2) = 0.0;      T.at<double>(2, 3) = -1.0;
//        T.at<double>(3, 0) = p2[a].x;  T.at<double>(3, 1) = 0.0;      T.at<double>(3, 2) = p1[a].x;  T.at<double>(3, 3) = 0.0;
//        T.at<double>(4, 0) = 0.0;      T.at<double>(4, 1) = p2[a].x;  T.at<double>(4, 2) = p1[a].y;  T.at<double>(4, 3) = 0.0;
//        T.at<double>(5, 0) = 0.0;      T.at<double>(5, 1) = 0.0;      T.at<double>(5, 2) = 1.0;      T.at<double>(5, 3) = 0.0;
//        T.at<double>(6, 0) = 0.0;      T.at<double>(6, 1) = 0.0;      T.at<double>(6, 2) = 0.0;      T.at<double>(6, 3) = 0.0;
//        T.at<double>(7, 0) = 0.0;      T.at<double>(7, 1) = 0.0;      T.at<double>(7, 2) = 0.0;      T.at<double>(7, 3) = 0.0;
//        T.at<double>(8, 0) = 0.0;      T.at<double>(8, 1) = 0.0;      T.at<double>(8, 2) = 0.0;      T.at<double>(8, 3) = 0.0;
        T.at<double>(0, 0) = -p2[a].y;
        T.at<double>(0, 3) = -p1[a].x;
        T.at<double>(1, 1) = -p2[a].y;                                
        T.at<double>(1, 3) = -p1[a].y;
        T.at<double>(2, 3) = -1.0;
        T.at<double>(3, 0) = p2[a].x;                                 
        T.at<double>(3, 2) = p1[a].x;
        T.at<double>(4, 1) = p2[a].x;  
        T.at<double>(4, 2) = p1[a].y;
        T.at<double>(5, 2) = 1.0;
        break;
    }
    return T;
}

//cv::Mat→Eigen::MatrixXd
Eigen::MatrixXd CvM2EgM(cv::Mat& X)
{
    int rw = X.rows;
    int cl = X.cols;
    Eigen::MatrixXd A(rw, cl);
    for (int i = 0; i < rw; i++)
        for (int j = 0; j < cl; j++) {
            A(i, j) = X.at<double>(i, j);
        }
    return A;
}
//Eigen::MatrixXd→cv::Mat
cv::Mat EgM2CvM(Eigen::MatrixXd& A)
{
    int rw = A.rows();
    int cl = A.cols();
    cv::Mat X = cv::Mat::zeros(rw, cl, CV_64FC1);
    for (int i = 0; i < rw; i++)
        for (int j = 0; j < cl; j++) {
            X.at<double>(i, j) = A(i, j);
        }
    return X;
}

//一般固有値問題について固有値λと固有ベクトルから最大固有値に対応する固有ベクトルθを計算する
cv::Mat Get_v(cv::Mat& M, cv::Mat& N)
{
    Eigen::MatrixXd m = CvM2EgM(M);
    Eigen::MatrixXd n = CvM2EgM(N);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(n, m); //一般化特異値分解
    Eigen::MatrixXd e = es.eigenvectors().col(8); // 8番目の固有ベクトルが最大
    e = e / e.norm(); //単位ベクトル化
    cv::Mat E = EgM2CvM(e);
    return E;
}

//行列Xのランクr制約一般逆行列
cv::Mat GenInv(cv::Mat& X)
{
    int r = 2; //ランク
    if (X.cols != X.rows) {
        std::cout << "正方行列ではありません" << std::endl;
        std::abort();
    }
    int n = X.cols; //サイズ
    Eigen::MatrixXd A = CvM2EgM(X);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ES(A);
    Eigen::MatrixXd GenI = Eigen::MatrixXd::Zero(9, 9);
    for (int i = n - r; i < n; i++)
        GenI += ES.eigenvectors().col(i) * ES.eigenvectors().col(i).transpose() / ES.eigenvalues()(i);
    cv::Mat B = EgM2CvM(GenI);
    return B;
}

//taubin法初期解x0
cv::Mat compute_x0(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2)
{
    int cl = (int)p1.size(); //残った対応点数
    cv::Mat M = cv::Mat::zeros(9, 9, CV_64FC1);
    cv::Mat N = M.clone();
    cv::Mat x0 = cv::Mat::zeros(9, 1, CV_64FC1);
    cv::Mat xn1 = x0.clone();
    for (int a = 0; a < cl; a++) {
        for (int k = 0; k < 3; k++) {
            cv::Mat ks = set_ks(p1, p2, a, k);
            cv::Mat T = set_T(p1, p2, a, k);
            M = M + ks * ks.t();
            N = N + T * T.t();
        }
    }
    M = M / cl;
    N = N / cl;
    x0 = Get_v(M, N);

    return x0;
}

//taubin法くりこみ演算xn1
cv::Mat hyper_renormalization(std::vector<cv::Point2d>& p1, std::vector<cv::Point2d>& p2, cv::Mat& x0)
{
    int cl = (int)p1.size();
    cv::Mat M, N;
    cv::Mat xn1 = x0.clone();
    cv::Mat W = cv::Mat::zeros(9, 9, CV_64FC1);
    cv::Mat V = cv::Mat::zeros(9, 9, CV_64FC1);
    cv::Mat v, ksk, ksl, Tk, Tl;
    double w, x_dot, dtheta, dtheta_copy;
    double Conv_eps = 1.0e-6; //1.0e-6
    int count_th = 6;
    int count = 0;
    while (1)
    {
        M = cv::Mat::zeros(9, 9, CV_64FC1);
        N = cv::Mat::zeros(9, 9, CV_64FC1);
        x0 = xn1.clone();
        for (int a = 0; a < cl; a++) {
            //重みを更新
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++) {
                    Tk = set_T(p1, p2, a, k);
                    Tl = set_T(p1, p2, a, l);
                    v = xn1.t() * Tk * Tl.t() * xn1;
                    V.at<double>(k, l) = v.at<double>(0, 0);
                }
            W = GenInv(V);
            //行列M,Nの計算
            for (int k = 0; k < 3; k++)
                for (int l = 0; l < 3; l++) {
                    ksk = set_ks(p1, p2, a, k);
                    ksl = set_ks(p1, p2, a, l);
                    Tk = set_T(p1, p2, a, k);
                    Tl = set_T(p1, p2, a, l);
                    w = W.at<double>(k, l);
                    M = M + w * ksk * ksl.t();
                    N = N + w * Tk * Tl.t();
                }
        }
        M /= cl;
        N /= cl;
        xn1 = Get_v(M, N);
        /*収束処理*/
        x_dot = x0.dot(xn1); //x0とxn1の内積
        if (x_dot < 0) xn1 = xn1 * (-1.0);
        dtheta = norm(xn1 - x0);
        std::cout << " |dtheta|=" << dtheta << std::endl;
        if (count < count_th)
            if (dtheta < Conv_eps)
                break;
            else {}
        else {
            std::cout << "xn1 = " << xn1 << std::endl;
            if (dtheta > dtheta_copy)
                break;
        }
        dtheta_copy = dtheta;
        count++;
    }
    return xn1;
}
/*
bool taubin(std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform)
{
    bool status = true;
    cv::Mat x0 = compute_x0(selectedtd, selectedcm);
    cv::Mat xn1 = hyper_renormalization(selectedtd, selectedcm, x0);
    tform = cnv_vc2mt(xn1);
    status = checkFunc(tform);
    return status;
}
*/
/*
bool sac::matrixestimation(const methods& ms, std::vector<cv::Point2d>& selectedcm, std::vector<cv::Point2d>& selectedtd, cv::Mat& tform)
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
    case matrixcalType::cTAUBIN:
        x0 = compute_x0(selectedtd, selectedcm);
        xn1 = hyper_renormalization(selectedtd, selectedcm, x0);
        tform = cnv_vc2mt(xn1);
        break;
    }
    status = checkFunc(tform);
    return status;
}
*/