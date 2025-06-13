#include "include/common.hpp"
#include <regex>
#include <sstream>
#include <queue>

namespace tradeknowledge {

class FastTextSearch {
private:
    // Boyer-Moore-Horspool algorithm for fast string matching
    std::vector<int> buildBadCharTable(const std::string& pattern) {
        std::vector<int> table(256, pattern.length());
        
        for (size_t i = 0; i < pattern.length() - 1; ++i) {
            table[static_cast<unsigned char>(pattern[i])] = pattern.length() - 1 - i;
        }
        
        return table;
    }
    
public:
    // Fast exact string search using Boyer-Moore-Horspool
    std::vector<size_t> search(const std::string& text, 
                               const std::string& pattern,
                               bool case_sensitive = true) {
        if (pattern.empty() || text.empty() || pattern.length() > text.length()) {
            return {};
        }
        
        std::string search_text = case_sensitive ? text : toLowerCase(text);
        std::string search_pattern = case_sensitive ? pattern : toLowerCase(pattern);
        
        auto badCharTable = buildBadCharTable(search_pattern);
        std::vector<size_t> matches;
        
        size_t i = 0;
        while (i <= search_text.length() - search_pattern.length()) {
            size_t j = search_pattern.length() - 1;
            
            while (j < search_pattern.length() && 
                   search_text[i + j] == search_pattern[j]) {
                if (j == 0) {
                    matches.push_back(i);
                    break;
                }
                --j;
            }
            
            i += badCharTable[static_cast<unsigned char>(search_text[i + search_pattern.length() - 1])];
        }
        
        return matches;
    }
    
    // Multi-pattern search using Aho-Corasick algorithm
    class AhoCorasick {
    private:
        struct Node {
            std::unordered_map<char, std::unique_ptr<Node>> children;
            std::vector<int> outputs;
            Node* failure = nullptr;
        };
        
        std::unique_ptr<Node> root;
        std::vector<std::string> patterns;
        
    public:
        AhoCorasick() : root(std::make_unique<Node>()) {}
        
        void addPattern(const std::string& pattern, int id) {
            patterns.push_back(pattern);
            Node* current = root.get();
            
            for (char c : pattern) {
                if (current->children.find(c) == current->children.end()) {
                    current->children[c] = std::make_unique<Node>();
                }
                current = current->children[c].get();
            }
            
            current->outputs.push_back(id);
        }
        
        void build() {
            // Build failure links using BFS
            std::queue<Node*> queue;
            
            // Initialize first level
            for (auto& [c, child] : root->children) {
                child->failure = root.get();
                queue.push(child.get());
            }
            
            // Build rest of the failure links
            while (!queue.empty()) {
                Node* current = queue.front();
                queue.pop();
                
                for (auto& [c, child] : current->children) {
                    queue.push(child.get());
                    
                    Node* failure = current->failure;
                    while (failure && failure->children.find(c) == failure->children.end()) {
                        failure = failure->failure;
                    }
                    
                    if (failure) {
                        child->failure = failure->children[c].get();
                        // Merge outputs
                        child->outputs.insert(child->outputs.end(),
                                            child->failure->outputs.begin(),
                                            child->failure->outputs.end());
                    } else {
                        child->failure = root.get();
                    }
                }
            }
        }
        
        std::vector<std::pair<size_t, int>> search(const std::string& text) {
            std::vector<std::pair<size_t, int>> matches;
            Node* current = root.get();
            
            for (size_t i = 0; i < text.length(); ++i) {
                char c = text[i];
                
                while (current != root.get() && 
                       current->children.find(c) == current->children.end()) {
                    current = current->failure;
                }
                
                if (current->children.find(c) != current->children.end()) {
                    current = current->children[c].get();
                }
                
                for (int id : current->outputs) {
                    size_t pos = i - patterns[id].length() + 1;
                    matches.push_back({pos, id});
                }
            }
            
            return matches;
        }
    };
    
    // Fuzzy search using edit distance
    int levenshteinDistance(const std::string& s1, const std::string& s2) {
        const size_t len1 = s1.size(), len2 = s2.size();
        std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));
        
        for (size_t i = 0; i <= len1; ++i) dp[i][0] = i;
        for (size_t j = 0; j <= len2; ++j) dp[0][j] = j;
        
        for (size_t i = 1; i <= len1; ++i) {
            for (size_t j = 1; j <= len2; ++j) {
                int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
                dp[i][j] = std::min({
                    dp[i-1][j] + 1,      // deletion
                    dp[i][j-1] + 1,      // insertion
                    dp[i-1][j-1] + cost  // substitution
                });
            }
        }
        
        return dp[len1][len2];
    }
    
    // Parallel search across multiple texts
    std::vector<std::pair<int, std::vector<size_t>>> 
    parallelSearch(const std::vector<std::string>& texts,
                   const std::string& pattern,
                   int num_threads = 0) {
        if (num_threads <= 0) {
            num_threads = omp_get_max_threads();
        }
        
        std::vector<std::pair<int, std::vector<size_t>>> all_results(texts.size());
        
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < texts.size(); ++i) {
            auto matches = search(texts[i], pattern, false);
            if (!matches.empty()) {
                all_results[i] = {static_cast<int>(i), matches};
            }
        }
        
        // Filter out empty results
        all_results.erase(
            std::remove_if(all_results.begin(), all_results.end(),
                          [](const auto& p) { return p.second.empty(); }),
            all_results.end()
        );
        
        return all_results;
    }
};

} // namespace tradeknowledge