#include "include/common.hpp"
#include <regex>
#include <sstream>

namespace tradeknowledge {

class FastTokenizer {
private:
    std::regex word_pattern;
    std::regex sentence_pattern;
    
public:
    FastTokenizer() {
        // Initialize regex patterns
        word_pattern = std::regex(R"(\b\w+\b)");
        sentence_pattern = std::regex(R"([.!?]+\s+)");
    }
    
    // Tokenize text into words
    StringVec tokenizeWords(const std::string& text) {
        StringVec tokens;
        
        std::sregex_iterator iter(text.begin(), text.end(), word_pattern);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            tokens.push_back(iter->str());
        }
        
        return tokens;
    }
    
    // Tokenize text into sentences
    StringVec tokenizeSentences(const std::string& text) {
        StringVec sentences;
        
        std::sregex_token_iterator iter(text.begin(), text.end(), sentence_pattern, -1);
        std::sregex_token_iterator end;
        
        for (; iter != end; ++iter) {
            std::string sentence = iter->str();
            if (!sentence.empty() && sentence.find_first_not_of(" \t\n\r") != std::string::npos) {
                sentences.push_back(sentence);
            }
        }
        
        return sentences;
    }
    
    // Simple whitespace tokenization (fastest)
    StringVec tokenizeWhitespace(const std::string& text) {
        StringVec tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        return tokens;
    }
    
    // N-gram generation
    StringVec generateNgrams(const StringVec& tokens, int n) {
        StringVec ngrams;
        
        if (tokens.size() < static_cast<size_t>(n)) {
            return ngrams;
        }
        
        for (size_t i = 0; i <= tokens.size() - n; ++i) {
            std::string ngram;
            for (int j = 0; j < n; ++j) {
                if (j > 0) ngram += " ";
                ngram += tokens[i + j];
            }
            ngrams.push_back(ngram);
        }
        
        return ngrams;
    }
    
    // Clean and normalize text
    std::string cleanText(const std::string& text) {
        std::string cleaned = text;
        
        // Convert to lowercase
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
        
        // Remove extra whitespace
        cleaned = std::regex_replace(cleaned, std::regex(R"(\s+)"), " ");
        
        // Trim
        cleaned.erase(0, cleaned.find_first_not_of(" \t\n\r"));
        cleaned.erase(cleaned.find_last_not_of(" \t\n\r") + 1);
        
        return cleaned;
    }
};

} // namespace tradeknowledge