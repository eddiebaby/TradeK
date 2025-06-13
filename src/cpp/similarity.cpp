#include "include/common.hpp"
#include <cstring>

namespace tradeknowledge {

class SimdSimilarity {
public:
    // Helper function to sum vector elements (fallback for when SIMD not available)
    float horizontalSum(const FloatVec& vec) {
        float sum = 0.0f;
        for (float val : vec) {
            sum += val;
        }
        return sum;
    }
    
    // Cosine similarity (standard implementation)
    float cosineSimilarity(const FloatVec& vec1, const FloatVec& vec2) {
        if (vec1.size() != vec2.size() || vec1.empty()) {
            return 0.0f;
        }
        
        float dot_product = 0.0f;
        float norm1_sq = 0.0f;
        float norm2_sq = 0.0f;
        
        for (size_t i = 0; i < vec1.size(); ++i) {
            dot_product += vec1[i] * vec2[i];
            norm1_sq += vec1[i] * vec1[i];
            norm2_sq += vec2[i] * vec2[i];
        }
        
        float norm_product = std::sqrt(norm1_sq * norm2_sq);
        if (norm_product < EPSILON) {
            return 0.0f;
        }
        
        return dot_product / norm_product;
    }
    
    // Euclidean distance
    float euclideanDistance(const FloatVec& vec1, const FloatVec& vec2) {
        if (vec1.size() != vec2.size() || vec1.empty()) {
            return std::numeric_limits<float>::max();
        }
        
        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < vec1.size(); ++i) {
            float diff = vec1[i] - vec2[i];
            sum_sq_diff += diff * diff;
        }
        
        return std::sqrt(sum_sq_diff);
    }
    
    // Manhattan distance
    float manhattanDistance(const FloatVec& vec1, const FloatVec& vec2) {
        if (vec1.size() != vec2.size() || vec1.empty()) {
            return std::numeric_limits<float>::max();
        }
        
        float sum_abs_diff = 0.0f;
        for (size_t i = 0; i < vec1.size(); ++i) {
            sum_abs_diff += std::abs(vec1[i] - vec2[i]);
        }
        
        return sum_abs_diff;
    }
    
    // Batch similarity computation
    std::vector<float> batchCosineSimilarity(const std::vector<FloatVec>& vectors,
                                           const FloatVec& query) {
        std::vector<float> similarities;
        similarities.reserve(vectors.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < vectors.size(); ++i) {
            float sim = cosineSimilarity(vectors[i], query);
            #pragma omp critical
            {
                similarities.push_back(sim);
            }
        }
        
        return similarities;
    }
    
    // Find top K most similar vectors
    std::vector<std::pair<int, float>> topKSimilar(const std::vector<FloatVec>& vectors,
                                                  const FloatVec& query,
                                                  int k) {
        std::vector<std::pair<int, float>> similarities;
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            float sim = cosineSimilarity(vectors[i], query);
            similarities.push_back({static_cast<int>(i), sim});
        }
        
        // Partial sort to get top K
        std::nth_element(similarities.begin(), 
                        similarities.begin() + k,
                        similarities.end(),
                        [](const auto& a, const auto& b) {
                            return a.second > b.second;
                        });
        
        similarities.resize(k);
        std::sort(similarities.begin(), similarities.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });
        
        return similarities;
    }
};

} // namespace tradeknowledge