#!/bin/bash
# Build script for C++ extensions

echo "Building C++ extensions for TradeKnowledge..."

# Check if pybind11 is installed
python -c "import pybind11" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: pybind11 not found. Installing..."
    pip install pybind11
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -name "*.so" -delete

# Create minimal C++ files if they don't exist
mkdir -p src/cpp

if [ ! -f "src/cpp/bindings.cpp" ]; then
    echo "Creating minimal C++ bindings..."
    cat > src/cpp/bindings.cpp << 'EOF'
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

std::string fast_search(const std::string& text, const std::string& query) {
    // Placeholder for fast search implementation
    return "Fast search not yet implemented";
}

std::vector<float> calculate_similarity(const std::vector<std::string>& texts) {
    // Placeholder for similarity calculation
    std::vector<float> scores(texts.size(), 0.5f);
    return scores;
}

PYBIND11_MODULE(tradeknowledge_cpp, m) {
    m.doc() = "TradeKnowledge C++ performance modules";
    m.def("fast_search", &fast_search, "Fast text search");
    m.def("calculate_similarity", &calculate_similarity, "Calculate text similarity");
}
EOF
fi

# Create other minimal files
for file in text_search.cpp similarity.cpp tokenizer.cpp; do
    if [ ! -f "src/cpp/$file" ]; then
        echo "// Placeholder for $file" > "src/cpp/$file"
    fi
done

# Build the extension
echo "Building C++ extensions..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✅ C++ extensions built successfully!"
    
    # Test the module
    python -c "
try:
    import tradeknowledge_cpp
    print('✅ C++ module imports successfully')
    print('Available functions:', dir(tradeknowledge_cpp))
except ImportError as e:
    print('❌ Failed to import C++ module:', e)
"
else
    echo "❌ Build failed. C++ extensions will be disabled."
    echo "The system will fall back to pure Python implementations."
fi