#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class FlannLshMatcher {
private:
    cv::Ptr<cv::FlannBasedMatcher> matcher_;
    float ratio_thresh_; // Loweのレシオテストで使用する閾値
    int knn_k_;          // k近傍探索で使用するkの値 (レシオテストには通常2が必要)

public:
    /**
     * @brief FlannLshMatcherのコンストラクタ。
     * @param ratio_thresh レシオテストの閾値。デフォルトは0.75f。
     * @param k k近傍探索で使用するkの値。レシオテストのためには2以上を推奨。デフォルトは2。
     */
    FlannLshMatcher(float ratio_thresh = 0.8f, int k = 2);

    /**
     * @brief FLANNベースのLSHを使用して特徴量マッチングを行います。
     * インターフェースはCustomLshMatcherと共通です。
     * @param oArea クエリ画像（主にデバッグ描画用）。
     * @param oPts クエリ画像から抽出されたキーポイント。
     * @param oFeatures クエリ画像から抽出された特徴記述子 (CV_8Uを期待)。
     * @param tArea ターゲット（データベース/訓練）画像（主にデバッグ描画用）。
     * @param tPts ターゲット画像から抽出されたキーポイント。
     * @param tFeatures ターゲット画像から抽出された特徴記述子 (CV_8Uを期待)。
     * @param oPts_matched マッチしたクエリキーポイントの出力先。
     * @param oFeatures_matched マッチしたクエリ特徴記述子の出力先。
     * @param tPts_matched マッチしたターゲットキーポイントの出力先。
     * @param tFeatures_matched マッチしたターゲット特徴記述子の出力先。
     * @return マッチしたペアの数。
     */
    size_t match_normal(
        cv::Mat& oArea, std::vector<cv::KeyPoint>& oPts, cv::Mat& oFeatures,
        cv::Mat& tArea, std::vector<cv::KeyPoint>& tPts, cv::Mat& tFeatures,
        std::vector<cv::KeyPoint>& oPts_matched, cv::Mat& oFeatures_matched,
        std::vector<cv::KeyPoint>& tPts_matched, cv::Mat& tFeatures_matched
    );
};