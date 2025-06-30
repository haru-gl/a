#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <random>
#include <cstdint> // For uint8_t

class CustomLshMatcher { // クラス名を CustomLshMatcher に変更
private:
    int knn_k;
    float knn_matchratio;
    int n_bits;    // 1テーブルあたりのbit数
    int n_tables;  // LSHテーブル数
    std::vector<std::vector<int>> hash_bits;
    std::vector<std::unordered_map<size_t, std::vector<int>>> tables;
    std::vector<std::vector<uint8_t>> db_descs; // tFeaturesをコピー

    // 静的メソッドはそのまま (実装は .cpp ファイル内)
    static size_t get_lsh_hash(const std::vector<uint8_t>& desc, const std::vector<int>& bits);
    static int hamming_distance(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);
    static std::vector<std::vector<uint8_t>> mat_to_vec(const cv::Mat& mat);

    // 重複排除メソッドの宣言
    void duplication_check_on_final_matches(std::vector<cv::DMatch>& matches);

public:
    // コンストラクタ名をクラス名に合わせる
    CustomLshMatcher(int knn_k_ = 2, float knn_matchratio_ = 0.6f, int n_bits_ = 12, int n_tables_ = 10);

    // メソッドシグネチャは以前の議論の通り、他のマッチャーと合わせる
    size_t match_normal(
        cv::Mat& oArea, std::vector<cv::KeyPoint>& oPts, cv::Mat& oFeatures,
        cv::Mat& tArea, std::vector<cv::KeyPoint>& tPts, cv::Mat& tFeatures,
        std::vector<cv::KeyPoint>& oPts_matched, cv::Mat& oFeatures_matched,
        std::vector<cv::KeyPoint>& tPts_matched, cv::Mat& tFeatures_matched
    );
};