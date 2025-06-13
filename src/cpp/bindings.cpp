#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/common.hpp"

// Include the actual implementations
#include "text_search.cpp"
#include "similarity.cpp"
#include "tokenizer.cpp"

namespace py = pybind11;
using namespace tradeknowledge;

PYBIND11_MODULE(tradeknowledge_cpp, m) {
    m.doc() = "TradeKnowledge C++ performance modules";
    
    // FastTextSearch class
    py::class_<FastTextSearch>(m, "FastTextSearch")
        .def(py::init<>())
        .def("search", &FastTextSearch::search, 
             "Fast string search using Boyer-Moore-Horspool",
             py::arg("text"), py::arg("pattern"), py::arg("case_sensitive") = true)
        .def("levenshtein_distance", &FastTextSearch::levenshteinDistance,
             "Calculate Levenshtein distance between two strings")
        .def("parallel_search", &FastTextSearch::parallelSearch,
             "Parallel search across multiple texts",
             py::arg("texts"), py::arg("pattern"), py::arg("num_threads") = 0);
    
    // SimdSimilarity class  
    py::class_<SimdSimilarity>(m, "SimdSimilarity")
        .def(py::init<>())
        .def("cosine_similarity", &SimdSimilarity::cosineSimilarity,
             "Calculate cosine similarity between two vectors")
        .def("euclidean_distance", &SimdSimilarity::euclideanDistance,
             "Calculate Euclidean distance between two vectors")
        .def("manhattan_distance", &SimdSimilarity::manhattanDistance,
             "Calculate Manhattan distance between two vectors")
        .def("batch_cosine_similarity", &SimdSimilarity::batchCosineSimilarity,
             "Batch cosine similarity computation")
        .def("top_k_similar", &SimdSimilarity::topKSimilar,
             "Find top K most similar vectors",
             py::arg("vectors"), py::arg("query"), py::arg("k"));
    
    // FastTokenizer class
    py::class_<FastTokenizer>(m, "FastTokenizer")
        .def(py::init<>())
        .def("tokenize_words", &FastTokenizer::tokenizeWords,
             "Tokenize text into words")
        .def("tokenize_sentences", &FastTokenizer::tokenizeSentences,
             "Tokenize text into sentences")
        .def("tokenize_whitespace", &FastTokenizer::tokenizeWhitespace,
             "Simple whitespace tokenization")
        .def("generate_ngrams", &FastTokenizer::generateNgrams,
             "Generate n-grams from tokens",
             py::arg("tokens"), py::arg("n"))
        .def("clean_text", &FastTokenizer::cleanText,
             "Clean and normalize text");
    
    // Utility functions
    m.def("to_lowercase", &toLowerCase, "Convert string to lowercase");
}