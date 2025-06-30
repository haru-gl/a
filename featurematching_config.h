#pragma once

enum class InitialMatcherType {
    KNN,          // 従来のk-NNを使用
    CUSTOM_LSH,   // 自作LSH (CustomLshMatcher)
    FLANN_LSH     // FLANNベースLSH (FlannLshMatcher)
};