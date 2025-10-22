#!/bin/bash

# Run All Tests for MoE-Plugin Pipeline
# Chạy tất cả tests để kiểm tra pipeline

echo "🧪 MoE-Plugin Pipeline Tests"
echo "============================"

# Test individual stages
echo ""
echo "🔍 Testing Individual Stages..."
echo "=============================="

# Stage 1 tests
echo ""
echo "🧪 Testing Stage 1..."
python test_stage1.py
if [ $? -eq 0 ]; then
    echo "✅ Stage 1 tests passed!"
else
    echo "❌ Stage 1 tests failed!"
    exit 1
fi

# Stage 2 tests
echo ""
echo "🧪 Testing Stage 2..."
python test_stage2.py
if [ $? -eq 0 ]; then
    echo "✅ Stage 2 tests passed!"
else
    echo "❌ Stage 2 tests failed!"
    exit 1
fi

# Stage 3 tests
echo ""
echo "🧪 Testing Stage 3..."
python test_stage3.py
if [ $? -eq 0 ]; then
    echo "✅ Stage 3 tests passed!"
else
    echo "❌ Stage 3 tests failed!"
    exit 1
fi

# Full pipeline tests
echo ""
echo "🔍 Testing Full Pipeline..."
echo "============================"

python test_full_pipeline.py
if [ $? -eq 0 ]; then
    echo "✅ Full pipeline tests passed!"
else
    echo "❌ Full pipeline tests failed!"
    exit 1
fi

echo ""
echo "🎉 All tests completed successfully!"
echo "✅ MoE-Plugin Pipeline is ready to run!"
echo ""
echo "📋 Next Steps:"
echo "1. Run full pipeline: python run_full_pipeline.py"
echo "2. Or run individual stages: ./run_stage1.sh, ./run_stage2.sh, ./run_stage3.sh"
echo "3. Check results in results/ directory"