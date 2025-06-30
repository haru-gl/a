#include "custom_lsh_matcher.h" // ヘッダーファイル名を変更
#include <algorithm>            // std::sort, std::min, std::unique
#include <vector>               // std::vector
#include <unordered_set>        // std::unordered_set
#include <unordered_map>        // std::unordered_map
#include <random>               // std::mt19937, std::uniform_int_distribution
#include <ctime>                // std::time
#include <cstring>              // std::memcpy
#include <iostream>             // std::cout, std::cerr

// ハミング距離計算用のポップカウントテーブル (ファイルスコープで静的)
static const int popcount_table[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

// クラス名を CustomLshMatcher に変更
CustomLshMatcher::CustomLshMatcher(int knn_k_, float knn_matchratio_, int n_bits_, int n_tables_)
    : knn_k(knn_k_), knn_matchratio(knn_matchratio_), n_bits(n_bits_), n_tables(n_tables_) {
    std::mt19937 rng((unsigned)std::time(nullptr));
    // AKAZEの記述子は通常61バイト (488ビット) だが、OpenCVでは可変長や固定長 (32, 64バイトなど) もあり得る。
    // 元のコードは256ビット (32バイト) を前提としているコメントがあったため、それに従う。
    // 記述子のバイト長が32バイトの場合、ビットインデックスは 0 から 32*8 - 1 = 255 まで。
    std::uniform_int_distribution<int> dist(0, 255); // 32バイト記述子の場合
    hash_bits.resize(n_tables);
    for (auto& bits_for_table : hash_bits) {
        bits_for_table.clear();
        for (int i = 0; i < n_bits; ++i) {
            bits_for_table.push_back(dist(rng));
        }
    }
}

// 静的メソッドの実装 (クラススコープ解決子を変更)
std::vector<std::vector<uint8_t>> CustomLshMatcher::mat_to_vec(const cv::Mat& mat) {
    std::vector<std::vector<uint8_t>> vec(mat.rows, std::vector<uint8_t>(mat.cols));
    for (int i = 0; i < mat.rows; ++i) {
        if (mat.ptr(i) && mat.cols > 0) {
            std::memcpy(vec[i].data(), mat.ptr(i), mat.cols);
        }
    }
    return vec;
}

size_t CustomLshMatcher::get_lsh_hash(const std::vector<uint8_t>& desc_param, const std::vector<int>& bits_param) {
    size_t hash = 0;
    if (desc_param.empty()) return hash;

    for (size_t i = 0; i < bits_param.size(); ++i) {
        int bit_index = bits_param[i];
        int byte_pos = bit_index / 8;
        int bit_pos_in_byte = bit_index % 8;

        // バイト位置が記述子の範囲内か確認
        if (static_cast<size_t>(byte_pos) < desc_param.size()) {
            hash |= (((desc_param[byte_pos] >> bit_pos_in_byte) & 1) << i);
        }
        // 範囲外の場合、そのビットは0として扱われる (hashにORされない)
    }
    return hash;
}

int CustomLshMatcher::hamming_distance(const std::vector<uint8_t>& a_desc, const std::vector<uint8_t>& b_desc) {
    int d = 0;
    size_t len = std::min(a_desc.size(), b_desc.size());
    for (size_t i = 0; i < len; ++i) {
        d += popcount_table[a_desc[i] ^ b_desc[i]];
    }
    // 長さが異なる場合、差分を距離に加算することも考慮できる (オプション)
    // d += std::abs(static_cast<int>(a_desc.size()) - static_cast<int>(b_desc.size())) * 8;
    return d;
}

void CustomLshMatcher::duplication_check_on_final_matches(std::vector<cv::DMatch>& matches) {
    if (matches.empty()) return;

    // 1. trainIdx でソートし、同じ trainIdx の中では distance でソート
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        if (a.trainIdx != b.trainIdx) {
            return a.trainIdx < b.trainIdx;
        }
        return a.distance < b.distance;
        });

    // 2. ユニークな trainIdx のみを保持 (同じ trainIdx の場合は最初のもの=distance最小が残る)
    matches.erase(
        std::unique(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.trainIdx == b.trainIdx;
            }),
        matches.end()
                );
}

size_t CustomLshMatcher::match_normal(
    cv::Mat& oArea, std::vector<cv::KeyPoint>& oPts, cv::Mat& oFeatures,
    cv::Mat& tArea, std::vector<cv::KeyPoint>& tPts, cv::Mat& tFeatures,
    std::vector<cv::KeyPoint>& oPts_matched, cv::Mat& oFeatures_matched,
    std::vector<cv::KeyPoint>& tPts_matched, cv::Mat& tFeatures_matched
) {
    if (oFeatures.empty() || tFeatures.empty()) {
        std::cerr << "CustomLshMatcher Error: Query (oFeatures) or Train (tFeatures) descriptors are empty." << std::endl;
        return 0;
    }
    if (oFeatures.type() != CV_8U || tFeatures.type() != CV_8U) {
        std::cerr << "CustomLshMatcher Error: Feature descriptors must be of type CV_8U." << std::endl;
        return 0;
    }

    auto query_descs = mat_to_vec(oFeatures); // oFeatures を使用
    db_descs = mat_to_vec(tFeatures);       // tFeatures を使用

    tables.clear();
    tables.resize(n_tables);
    for (size_t idx = 0; idx < db_descs.size(); ++idx) {
        if (db_descs[idx].empty()) continue; // 空の記述子はスキップ
        for (int t = 0; t < n_tables; ++t) {
            size_t hash = get_lsh_hash(db_descs[idx], hash_bits[t]);
            tables[t][hash].push_back(static_cast<int>(idx));
        }
    }

    std::vector<cv::DMatch> passed_ratio_test_matches;

    for (size_t qi = 0; qi < query_descs.size(); ++qi) {
        if (query_descs[qi].empty()) continue;

        // 1. voteスコア制用のマップ: 候補index → ヒット回数
        std::unordered_map<int, int> candidate_votes;
        for (int t = 0; t < n_tables; ++t) {
            size_t hash = get_lsh_hash(query_descs[qi], hash_bits[t]);
            auto it = tables[t].find(hash);
            if (it != tables[t].end()) {
                for (int idx : it->second) {
                    candidate_votes[idx]++;
                }
            }
        }

        // 2. vote数で降順ソート
        std::vector<std::pair<int, int>> vote_vec(candidate_votes.begin(), candidate_votes.end());
        std::sort(vote_vec.begin(), vote_vec.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        // 3. 上位N個だけを候補に（Nはknn_kより大きめにしてもOK。ここではmax(10, knn_k*2)にしてみる例）
        int max_candidates = std::max(10, knn_k * 2);
        if (vote_vec.size() > (size_t)max_candidates) {
            vote_vec.resize(max_candidates);
        }

        // 4. 上記候補の中から「ハミング距離が近い順」でknn_k件を抽出
        std::vector<std::pair<int, int>> dists; // <index, distance>
        for (const auto& pr : vote_vec) {
            int idx = pr.first;
            if (static_cast<size_t>(idx) < db_descs.size() && !db_descs[idx].empty()) {
                dists.emplace_back(idx, hamming_distance(query_descs[qi], db_descs[idx]));
            }
        }
        if (dists.empty()) continue;

        // 5. ハミング距離で昇順ソート、knn_k件だけ使う
        std::partial_sort(dists.begin(), dists.begin() + std::min(knn_k, (int)dists.size()), dists.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        // 6. レシオテスト
        if (dists.size() >= 2 && knn_matchratio < 1.0f) {
            if (dists[0].second < knn_matchratio * dists[1].second) {
                passed_ratio_test_matches.emplace_back(static_cast<int>(qi), dists[0].first, static_cast<float>(dists[0].second));
            }
        }
        else if (dists.size() == 1 && knn_k == 1) {
            passed_ratio_test_matches.emplace_back(static_cast<int>(qi), dists[0].first, static_cast<float>(dists[0].second));
        }
    }


    duplication_check_on_final_matches(passed_ratio_test_matches);

    oPts_matched.clear(); oFeatures_matched = cv::Mat();
    tPts_matched.clear(); tFeatures_matched = cv::Mat();

    for (const auto& match : passed_ratio_test_matches) {
        // queryIdx と trainIdx がそれぞれのキーポイント/特徴量の範囲内にあるか確認
        if (match.queryIdx >= 0 && static_cast<size_t>(match.queryIdx) < oPts.size() &&
            match.trainIdx >= 0 && static_cast<size_t>(match.trainIdx) < tPts.size()) {
            oPts_matched.push_back(oPts[match.queryIdx]);
            oFeatures_matched.push_back(oFeatures.row(match.queryIdx));
            tPts_matched.push_back(tPts[match.trainIdx]);
            tFeatures_matched.push_back(tFeatures.row(match.trainIdx));
        }
    }

    //std::cout << "CustomLshMatcher: Number of LSH matches after duplication check: " << oPts_matched.size() << std::endl;

    // 描画処理は、このマッチャーを使用する側 (例: featurematching_main.cpp) で行うことを推奨
    // もしこのクラス内で描画が必要な場合は、以下のコメントを解除・調整
    /*
    if (!oArea.empty() && !tArea.empty() && !passed_ratio_test_matches.empty()) {
        cv::Mat dmimg;
        cv::drawMatches(oArea, oPts, tArea, tPts, passed_ratio_test_matches, dmimg);
        cv::imshow("matches_custom_lsh", dmimg);
        cv::imwrite("results_custom_lsh.bmp", dmimg);
        // cv::waitKey(0); // アプリケーションのメインループで制御する場合
    }
    */

    return oPts_matched.size(); // マッチしたペア数を返す
}