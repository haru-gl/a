#include <utility>      // For std::swap (though <algorithm> is more common)
#include <algorithm>    // For std::swap if <utility> isn't sufficient
#include <vector>       // For std::vector
#include <opencv2/opencv.hpp>
#include "knn.h"
#include "classes.h"    // For clipedmap_data, target_data
#include "enclasses.h"  // For knnType enum

// ★ LSHマッチャーのヘッダーをインクルード
#include "custom_lsh_matcher.h"
#include "flann_lsh_matcher.h"

// matrow_swap 関数 (変更なし)
void matrow_swap(cv::Mat& des, int i, int j) { /* ... 元の実装 ... */
    unsigned char tmp;
    for (int k = 0; k < des.cols; k++) {
        tmp = des.at<unsigned char>(i, k);
        des.at<unsigned char>(i, k) = des.at<unsigned char>(j, k);
        des.at<unsigned char>(j, k) = tmp;
    }
}

// kp_sort 関数 (変更なし)
void kp_sort(std::vector<cv::Point2d>& cmppts, cv::Mat& cmpdes, std::vector<cv::Point2d>& tgtpts, cv::Mat& tgtdes, std::vector<float>& ratio) { /* ... 元の実装 ... */
    size_t num_kpts = cmppts.size();
    for (size_t i = 0; i < num_kpts - 1; i++)
        for (size_t j = i + 1; j < num_kpts; j++)
            if (ratio[i] > ratio[j]) {
                std::swap(ratio[i], ratio[j]);
                std::swap(cmppts[i], cmppts[j]);
                matrow_swap(cmpdes, (int)i, (int)j);
                std::swap(tgtpts[i], tgtpts[j]);
                matrow_swap(tgtdes, (int)i, (int)j);
            }
}


// knnクラスの他のメソッド (setters/getters) は変更なし
void knn::set_knn_k(int k) { knn_k = k; }
int knn::get_knn_k(void) const { return knn_k; }
void knn::set_knn_matchratio(double mr) { knn_matchratio = mr; }
double knn::get_knn_matchratio(void) const { return knn_matchratio; }
void knn::set_knn_sortflag(bool st) { knn_sort = st; }
bool knn::get_knn_sortflag(void) const { return knn_sort; }
void knn::set_knn_fixn(size_t maxn) { knn_fixn = maxn; }
size_t knn::get_knn_fixn(void) const { return knn_fixn; }
void knn::set_knn_spDist(double spd) { knn_spDist = spd; }
double knn::get_knn_spDist(void) const { return knn_spDist; }

// matchiratiocheck と detect_samepoint はk-NN専用のヘルパーなので、そのまま残す
size_t knn::matchiratiocheck(std::vector<std::vector<cv::DMatch>>& nn_matches, clipedmap_data& cm, target_data& td, std::vector<cv::Point2d>& cmpts, cv::Mat& cmdes, std::vector<cv::Point2d>& tgpts, cv::Mat& tgdes, std::vector<float>& ratio) { /* ... 元の実装 ... */
    size_t candnum = 0;
    tgpts.clear(); tgdes = cv::Mat(); cmpts.clear(); cmdes = cv::Mat(); ratio.clear();
    for (size_t n = 0; n < nn_matches.size(); n++) {
        if (nn_matches[n].size() < 2) continue; // 2つ以上の近傍がないとレシオテスト不可
        cv::DMatch first = nn_matches[n][0];
        if ((double)(nn_matches[n][0].distance) < knn_matchratio * (double)(nn_matches[n][1].distance)) {
            candnum++;
            tgpts.push_back((cv::Point2d)td.oPts[first.queryIdx].pt); // td.oPts を使用
            tgdes.push_back(td.oFeatures.row(first.queryIdx));      // td.oFeatures を使用
            cmpts.push_back((cv::Point2d)cm.oPts[first.trainIdx].pt); // cm.oPts を使用
            cmdes.push_back(cm.oFeatures.row(first.trainIdx));      // cm.oFeatures を使用
            ratio.push_back(fabs(nn_matches[n][0].distance / nn_matches[n][1].distance));
        }
    }
    return candnum;
}
size_t knn::detect_samepoint(std::vector<cv::Point2d>& pts, std::vector<float>& ratio) { /* ... 元の実装 ... */
    size_t candnum = pts.size();
    size_t rejc = 0;
    for (size_t n = 0; n < candnum; n++) {
        if (ratio[n] >= 1.0) continue;
        std::vector<size_t> smp;
        float minratio = ratio[n]; // 初期値を変更
        size_t mini = n;
        for (size_t i = n + 1; i < candnum; i++) {
            if (ratio[i] >= 1.0f) continue;
            if (cv::norm(pts[n] - pts[i]) < knn_spDist) {
                smp.push_back(i);
                if (minratio > ratio[i]) { // より小さいものを優先
                    minratio = ratio[i];
                    // mini = i; // このロジックは下で処理
                }
            }
        }
        if (smp.empty()) continue; // smp.size() == 0 から変更

        // n 自身とsmp内の他の点で、nが最小でなければnを棄却
        bool n_is_min_among_duplicates = true;
        for (size_t idx_in_smp : smp) {
            if (ratio[idx_in_smp] < ratio[n]) {
                n_is_min_among_duplicates = false;
                break;
            }
            else if (ratio[idx_in_smp] == ratio[n] && idx_in_smp < n) { // 同率ならインデックスが小さい方を優先
                n_is_min_among_duplicates = false;
                break;
            }
        }
        if (!n_is_min_among_duplicates) {
            ratio[n] = 1.1f; // n を棄却
            rejc++;
            // nが棄却されたので、smp内の処理は不要になる場合もあるが、
            // 他の要素がnとは別に重複している可能性もあるので、ループは継続
        }
        else { // n が重複グループ内で最小の場合、他を棄却
            for (size_t idx_in_smp : smp) {
                //if (smp[i] != mini) { // この条件は複雑だったので単純化
                if (idx_in_smp != n && (ratio[idx_in_smp] > ratio[n] || (ratio[idx_in_smp] == ratio[n] && idx_in_smp > n))) {
                    ratio[idx_in_smp] = 1.1f;
                    rejc++;
                }
            }
        }
        smp.clear(); smp.shrink_to_fit();
    }
    return rejc;
}


// ★ knn::match メソッドを修正
size_t knn::match(knnType kt, clipedmap_data& cmp, target_data& tgt) // cm は cmp に変更 (引数名)
{
    // このメソッドの出力は、cmp.oMatchedPts, cmp.oMatchedFeatures,
    // tgt.oMatchedPts, tgt.oMatchedFeatures と戻り値のマッチ数。

    // 最初に結果をクリア
    cmp.oMatchedPts.clear(); cmp.oMatchedFeatures = cv::Mat();
    tgt.oMatchedPts.clear(); tgt.oMatchedFeatures = cv::Mat();
    size_t goodPairsNum = 0;

    if (cmp.oFeatures.empty() || tgt.oFeatures.empty()) {
        std::cerr << "knn::match Error: Features are empty in map or target." << std::endl;
        return 0;
    }


    // LSHの場合の処理
    if (kt == knnType::kCUSTOM_LSH || kt == knnType::kFLANN_LSH) {
        std::vector<cv::KeyPoint> map_kpts_matched_lsh;
        cv::Mat map_desc_matched_lsh;
        std::vector<cv::KeyPoint> target_kpts_matched_lsh;
        cv::Mat target_desc_matched_lsh;

        if (kt == knnType::kCUSTOM_LSH) {
            // CustomLshMatcher のコンストラクタ引数に knn クラスのメンバを使用
            CustomLshMatcher matcher_custom(knn_k, static_cast<float>(knn_matchratio)); // n_bits, n_tablesはデフォルト
            goodPairsNum = matcher_custom.match_normal(
                cmp.oImage, cmp.oPts, cmp.oFeatures, // マップ側入力
                tgt.oImage, tgt.oPts, tgt.oFeatures, // ターゲット側入力
                map_kpts_matched_lsh, map_desc_matched_lsh, // マップ側出力 (KeyPoint, Mat)
                target_kpts_matched_lsh, target_desc_matched_lsh  // ターゲット側出力 (KeyPoint, Mat)
            );
        }
        else { // knnType::kFLANN_LSH
            // FlannLshMatcher のコンストラクタ引数に knn クラスのメンバを使用
            FlannLshMatcher matcher_flann(static_cast<float>(knn_matchratio), knn_k);
            goodPairsNum = matcher_flann.match_normal(
                cmp.oImage, cmp.oPts, cmp.oFeatures,
                tgt.oImage, tgt.oPts, tgt.oFeatures,
                map_kpts_matched_lsh, map_desc_matched_lsh,
                target_kpts_matched_lsh, target_desc_matched_lsh
            );
        }

        // LSHの結果 (KeyPoint) を Point2d に変換して cmp, tgt に格納
        for (const auto& kp : map_kpts_matched_lsh) cmp.oMatchedPts.push_back(kp.pt);
        cmp.oMatchedFeatures = map_desc_matched_lsh.clone();

        for (const auto& kp : target_kpts_matched_lsh) tgt.oMatchedPts.push_back(kp.pt);
        tgt.oMatchedFeatures = target_desc_matched_lsh.clone();

        std::cout << "knn::match (LSH Mode: " << kt << "): Initial matches = " << goodPairsNum << std::endl;
        return goodPairsNum; // LSHの場合はここで終了
    }

    // 以下は従来のk-NNの処理 (kt が kNORMAL, kDTSFP などの場合)
    cv::BFMatcher bf_matcher; // 元のコードではBFMatcherのインスタンスがローカルだった
    const bool isCrossCheck = false; // 元のコードから
    bf_matcher = cv::BFMatcher(knn_normType, isCrossCheck); // knn_normType はメンバ変数
    std::vector<std::vector<cv::DMatch>> nn_matches;
    bf_matcher.knnMatch(tgt.oFeatures, cmp.oFeatures, nn_matches, knn_k); // knn_k はメンバ変数

    std::vector<cv::Point2d> cmppts_temp, tgtpts_temp; // matchiratiocheck 用の一時変数
    cv::Mat cmpdes_temp, tgtdes_temp;
    std::vector<float> ratio_temp;
    size_t candnum = 0;
    // size_t rejt = 0, rejm = 0; // detect_samepoint用 (元のコードより)
    // size_t rfixn;             // kNNFIXN用 (元のコードより)

    // knnType::kNORMAL などの既存のk-NNロジック
    // (この部分は元の knn::match の実装をほぼそのまま流用)
    switch (kt) {
    case knnType::kNORMAL:
        // matchiratiocheck は cmp.oMatchedPts 等を直接設定しないので、一時変数で受ける
        candnum = matchiratiocheck(nn_matches, cmp, tgt, cmppts_temp, cmpdes_temp, tgtpts_temp, tgtdes_temp, ratio_temp);
        cmp.oMatchedPts = cmppts_temp;
        cmp.oMatchedFeatures = cmpdes_temp.clone();
        tgt.oMatchedPts = tgtpts_temp;
        tgt.oMatchedFeatures = tgtdes_temp.clone();
        goodPairsNum = candnum;
        // knn_sort は matchiratiocheck 内では処理されないので、ここで適用する場合
        if (knn_sort) kp_sort(cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, ratio_temp);
        break;
    case knnType::kDTSFP:
        candnum = matchiratiocheck(nn_matches, cmp, tgt, cmppts_temp, cmpdes_temp, tgtpts_temp, tgtdes_temp, ratio_temp);
        detect_samepoint(tgtpts_temp, ratio_temp); // ratio_temp を変更
        detect_samepoint(cmppts_temp, ratio_temp); // ratio_temp を変更

        for (size_t n = 0; n < candnum; n++) {
            if (ratio_temp[n] >= 1.0) continue;
            tgt.oMatchedPts.push_back(tgtpts_temp[n]);
            tgt.oMatchedFeatures.push_back(tgtdes_temp.row(static_cast<int>(n)));
            cmp.oMatchedPts.push_back(cmppts_temp[n]);
            cmp.oMatchedFeatures.push_back(cmpdes_temp.row(static_cast<int>(n)));
        }
        goodPairsNum = tgt.oMatchedPts.size();
        if (knn_sort) { // ソート対象は ratio_temp ではなく、実際に残ったマッチの順序
            std::vector<float> final_ratios; // 再構築が必要
            for (size_t n = 0; n < candnum; ++n) if (ratio_temp[n] < 1.0) final_ratios.push_back(ratio_temp[n]);
            kp_sort(cmp.oMatchedPts, cmp.oMatchedFeatures, tgt.oMatchedPts, tgt.oMatchedFeatures, final_ratios);
        }
        break;
    case knnType::kNNFIXN:
        candnum = matchiratiocheck(nn_matches, cmp, tgt, cmppts_temp, cmpdes_temp, tgtpts_temp, tgtdes_temp, ratio_temp);
        { // スコープ作成
            size_t rfixn_count = knn_fixn; // knn_fixn はメンバ変数
            if (rfixn_count > candnum) rfixn_count = candnum;

            std::vector<std::pair<float, size_t>> sorted_indices(candnum);
            for (size_t i = 0; i < candnum; ++i) sorted_indices[i] = { ratio_temp[i], i };
            std::sort(sorted_indices.begin(), sorted_indices.end());

            for (size_t n = 0; n < rfixn_count; n++) {
                size_t original_idx = sorted_indices[n].second;
                if (ratio_temp[original_idx] >= 1.0f) continue; // 既に無効なものはスキップ (通常はありえないが念のため)

                tgt.oMatchedPts.push_back(tgtpts_temp[original_idx]);
                tgt.oMatchedFeatures.push_back(tgtdes_temp.row(static_cast<int>(original_idx)));
                cmp.oMatchedPts.push_back(cmppts_temp[original_idx]);
                cmp.oMatchedFeatures.push_back(cmpdes_temp.row(static_cast<int>(original_idx)));
            }
            goodPairsNum = tgt.oMatchedPts.size();
            // kNNFIXNの場合、既にソートされた順で選んでいるのでknn_sortは不要かもしれないが、
            // もし元のkp_sortが別の基準（例：座標）でソートしているなら意味がある。
            // ここでは、ratioでソート済みなので、追加のknn_sortは不要と判断。
        }
        break;
    case knnType::kNNFIXNDTSFP:
        candnum = matchiratiocheck(nn_matches, cmp, tgt, cmppts_temp, cmpdes_temp, tgtpts_temp, tgtdes_temp, ratio_temp);
        detect_samepoint(tgtpts_temp, ratio_temp);
        detect_samepoint(cmppts_temp, ratio_temp);
        { // スコープ作成
            size_t rfixn_count = knn_fixn;
            std::vector<std::pair<float, size_t>> valid_matches_sorted;
            for (size_t i = 0; i < candnum; ++i) {
                if (ratio_temp[i] < 1.0f) {
                    valid_matches_sorted.push_back({ ratio_temp[i], i });
                }
            }
            std::sort(valid_matches_sorted.begin(), valid_matches_sorted.end());

            if (rfixn_count > valid_matches_sorted.size()) rfixn_count = valid_matches_sorted.size();

            for (size_t n = 0; n < rfixn_count; n++) {
                size_t original_idx = valid_matches_sorted[n].second;
                tgt.oMatchedPts.push_back(tgtpts_temp[original_idx]);
                tgt.oMatchedFeatures.push_back(tgtdes_temp.row(static_cast<int>(original_idx)));
                cmp.oMatchedPts.push_back(cmppts_temp[original_idx]);
                cmp.oMatchedFeatures.push_back(cmpdes_temp.row(static_cast<int>(original_idx)));
            }
            goodPairsNum = tgt.oMatchedPts.size();
        }
        break;
    default:
        std::cerr << "knn::match Error: Unknown knnType " << kt << std::endl;
        goodPairsNum = 0;
        break;
    }

    cmppts_temp.clear(); cmppts_temp.shrink_to_fit();
    tgtpts_temp.clear(); tgtpts_temp.shrink_to_fit();
    ratio_temp.clear(); ratio_temp.shrink_to_fit();
    tgtdes_temp = cv::Mat(); cmpdes_temp = cv::Mat();
    nn_matches.clear(); nn_matches.shrink_to_fit();

    std::cout << "knn::match (k-NN Mode: " << kt << "): Good pairs = " << goodPairsNum << std::endl;
    return goodPairsNum;
}