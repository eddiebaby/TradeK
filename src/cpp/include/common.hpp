#ifndef TRADEKNOWLEDGE_COMMON_HPP
#define TRADEKNOWLEDGE_COMMON_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace tradeknowledge {

// Type aliases for clarity
using StringVec = std::vector<std::string>;
using FloatVec = std::vector<float>;
using DoubleVec = std::vector<double>;
using IntVec = std::vector<int>;

// Constants
constexpr int DEFAULT_BATCH_SIZE = 1000;
constexpr float EPSILON = 1e-8f;

// Utility functions
inline std::string toLowerCase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

} // namespace tradeknowledge

#endif // TRADEKNOWLEDGE_COMMON_HPP