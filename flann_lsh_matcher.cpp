#include "flann_lsh_matcher.h"
#include <iostream> // std::cout, std::cerr
#include <vector>   // std::vector

FlannLshMatcher::FlannLshMatcher(float ratio_thresh, int k)
    : ratio_thresh_(ratio_thresh), knn_k_(k) {
    // AKAZEのようなバイナリ記述子 (CV_8U) にはLSHインデックスが適しています。
    // パラメータ (table_number, key_size, multi_probe_level) はデータセットや
    // 性能要件に応じて調整が必要な場合があります。
    // 例: cv::flann::LshIndexParams(int table_number, int key_size, int multi_probe_level)
    // table_number: ハッシュテーブルの数 (例: 6-20)
    // key_size: ハッシュキーのビット長 (例: 10-20)
    // multi_probe_level: 探索する近傍バケットの数を増やす (例: 1-2)
    matcher_ = cv::makePtr<cv::FlannBasedMatcher>(
        cv::makePtr<cv::flann::LshIndexParams>(20, 15, 0)// これらの値は一例(6, 12, 1) (8, 16, 1) (12, 16, 2)(12, 20, 2) 
    );
    // もしmatcher_が cv::DescriptorMatcher::create("FlannBased") のようにして
    // 作成される場合、set日本のLSHIndexParamsをYAMLなどから読み込ませることも可能です。
}

size_t FlannLshMatcher::match_normal(
    cv::Mat& oArea, std::vector<cv::KeyPoint>& oPts, cv::Mat& oFeatures,
    cv::Mat& tArea, std::vector<cv::KeyPoint>& tPts, cv::Mat& tFeatures,
    std::vector<cv::KeyPoint>& oPts_matched, cv::Mat& oFeatures_matched,
    std::vector<cv::KeyPoint>& tPts_matched, cv::Mat& tFeatures_matched
) {
    if (oFeatures.empty() || tFeatures.empty()) {
        std::cerr << "FlannLshMatcher Error: Query (oFeatures) or Train (tFeatures) descriptors are empty." << std::endl;
        return 0;
    }
    // FLANN LSHはバイナリ記述子(CV_8U)を期待します (AKAZEのMLDBなど)
    if (oFeatures.type() != CV_8U || tFeatures.type() != CV_8U) {
        std::cerr << "FlannLshMatcher Error: Feature descriptors must be of type CV_8U." << std::endl;
        return 0;
    }
    // レシオテストを行うには knn_k_ が2以上である必要があります。
    if (knn_k_ < 2 && ratio_thresh_ < 1.0f) { // ratio_thresh_が1.0未満ならレシオテスト意図
        std::cerr << "FlannLshMatcher Warning: knn_k is " << knn_k_ << " but should be >= 2 for ratio test."
            << " Ratio test might be ineffective or skipped." << std::endl;
    }
    if (matcher_.empty()) {
        std::cerr << "FlannLshMatcher Error: FlannBasedMatcher is not initialized." << std::endl;
        return 0;
    }

    std::vector<std::vector<cv::DMatch>> knn_matches_list;
    // knnMatch(queryDescriptors, trainDescriptors, matches, k) を使用
    // oFeatures がクエリ記述子、tFeatures が訓練(ターゲット)記述子
    matcher_->knnMatch(oFeatures, tFeatures, knn_matches_list, knn_k_);

    std::vector<cv::DMatch> good_matches;
    if (ratio_thresh_ < 1.0f) { // レシオテストを実行する場合 (knn_k_ >= 2 も暗黙的に期待)
        for (size_t i = 0; i < knn_matches_list.size(); i++) {
            // knn_matches_list[i] は i番目のクエリ記述子に対するk個のマッチ候補
            if (knn_matches_list[i].size() >= 2) { // 少なくとも2つの近傍が見つかった場合
                if (knn_matches_list[i][0].distance < ratio_thresh_ * knn_matches_list[i][1].distance) {
                    good_matches.push_back(knn_matches_list[i][0]); // 最も良いマッチを採用
                }
            }
            else if (knn_matches_list[i].size() == 1 && knn_k_ == 1) {
                // k=1 のみ指定され、レシオテストが意図されていない場合。
                // または、k>=2 だが候補が1つしか見つからなかった場合。
                // ここでは、k=1でレシオテストなしのシナリオを許容するなら、このマッチを採用する。
                // ただし、ratio_thresh_ < 1.0f の条件外で処理する方が明確かもしれない。
                good_matches.push_back(knn_matches_list[i][0]);
            }
        }
    }
    else { // レシオテストを行わない場合 (例: ratio_thresh_ >= 1.0 または k=1 のみ)
        for (size_t i = 0; i < knn_matches_list.size(); i++) {
            if (!knn_matches_list[i].empty()) {
                good_matches.push_back(knn_matches_list[i][0]); // 単純に最初の(最も近い)マッチを採用
            }
        }
    }

    std::cout << "FlannLshMatcher: Number of matches after ratio test: " << good_matches.size() << std::endl;

    // マッチしたキーポイントと特徴量をクリアして再格納
    oPts_matched.clear();
    oFeatures_matched = cv::Mat();
    tPts_matched.clear();
    tFeatures_matched = cv::Mat();

    for (const auto& match : good_matches) {
        if (match.queryIdx >= 0 && static_cast<size_t>(match.queryIdx) < oPts.size() &&
            match.trainIdx >= 0 && static_cast<size_t>(match.trainIdx) < tPts.size()) {

            oPts_matched.push_back(oPts[match.queryIdx]);
            oFeatures_matched.push_back(oFeatures.row(match.queryIdx));
            tPts_matched.push_back(tPts[match.trainIdx]);
            tFeatures_matched.push_back(tFeatures.row(match.trainIdx));
        }
        else {
            std::cerr << "FlannLshMatcher Warning: Invalid match index. queryIdx=" << match.queryIdx
                << ", trainIdx=" << match.trainIdx << std::endl;
        }
    }

    std::cout << "FlannLshMatcher: Number of final matches populated: " << oPts_matched.size() << std::endl;

    // 描画処理 (オプション、呼び出し側で行うことを推奨)
    /*
    if (!oArea.empty() && !tArea.empty() && !good_matches.empty()) {
        cv::Mat dmimg;
        cv::drawMatches(oArea, oPts, tArea, tPts, good_matches, dmimg,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("matches_flann_lsh_via_normal_if", dmimg); // ウィンドウ名を変更
        cv::imwrite("results_flann_lsh_via_normal_if.bmp", dmimg); // 保存ファイル名を変更
        // cv::waitKey(0);
    }
    */

    return oPts_matched.size(); // マッチしたペアの数を返す
}