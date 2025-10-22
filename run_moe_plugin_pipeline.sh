#!/bin/bash

# MoE-Plugin Pipeline - Run from Root Directory
# Chạy toàn bộ pipeline từ root directory

echo "🚀 MoE-Plugin Pipeline"
echo "====================="
echo "📁 Running from root directory"
echo ""

# Parse arguments
STAGES="1 2 3"
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --stages)
            STAGES="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--stages 1,2,3] [--skip-tests]"
            echo "  --stages: Stages to run (default: 1 2 3)"
            echo "  --skip-tests: Skip test scripts"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "📋 Stages to run: $STAGES"
echo "🧪 Skip tests: $SKIP_TESTS"
echo "====================="

# Run test scripts if not skipped
if [ "$SKIP_TESTS" = false ]; then
    echo ""
    echo "🧪 Running test scripts..."
    
    for stage in $STAGES; do
        case $stage in
            1)
                echo ""
                echo "🔍 Testing Stage 1..."
                python moe_plugin_pipeline/test_stage1.py
                if [ $? -ne 0 ]; then
                    echo "❌ Stage 1 tests failed!"
                    exit 1
                fi
                echo "✅ Stage 1 tests passed!"
                ;;
            2)
                echo ""
                echo "🔍 Testing Stage 2..."
                python moe_plugin_pipeline/test_stage2.py
                if [ $? -ne 0 ]; then
                    echo "❌ Stage 2 tests failed!"
                    exit 1
                fi
                echo "✅ Stage 2 tests passed!"
                ;;
            3)
                echo ""
                echo "🔍 Testing Stage 3..."
                python moe_plugin_pipeline/test_stage3.py
                if [ $? -ne 0 ]; then
                    echo "❌ Stage 3 tests failed!"
                    exit 1
                fi
                echo "✅ Stage 3 tests passed!"
                ;;
        esac
    done
fi

# Run stages
for stage in $STAGES; do
    case $stage in
        1)
            echo ""
            echo "🚀 Stage 1: Training BalPoE Experts"
            echo "=================================="
            
            # Check if Stage 1 config exists
            if [ ! -f "moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json" ]; then
                echo "❌ Stage 1 config not found!"
                exit 1
            fi
            
            # Run Stage 1
            python moe_plugin_pipeline/stage1_train_balpoe.py --config moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json
            
            if [ $? -eq 0 ]; then
                echo "✅ Stage 1 completed successfully!"
            else
                echo "❌ Stage 1 failed!"
                exit 1
            fi
            ;;
        2)
            echo ""
            echo "🚀 Stage 2: Optimizing Plugin"
            echo "============================="
            
            # Check if Stage 2 config exists
            if [ ! -f "moe_plugin_pipeline/configs/plugin_optimization.json" ]; then
                echo "❌ Stage 2 config not found!"
                exit 1
            fi
            
            # Check if experts exist
            if [ ! -d "checkpoints/balpoe_experts/cifar100_ir100" ]; then
                echo "❌ Experts directory not found!"
                exit 1
            fi
            
            # Run Stage 2
            python moe_plugin_pipeline/stage2_optimize_plugin.py \
                --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
                --config moe_plugin_pipeline/configs/plugin_optimization.json
            
            if [ $? -eq 0 ]; then
                echo "✅ Stage 2 completed successfully!"
            else
                echo "❌ Stage 2 failed!"
                exit 1
            fi
            ;;
        3)
            echo ""
            echo "🚀 Stage 3: Evaluation và Comparison"
            echo "=================================="
            
            # Check if Stage 3 config exists
            if [ ! -f "moe_plugin_pipeline/configs/evaluation_config.json" ]; then
                echo "❌ Stage 3 config not found!"
                exit 1
            fi
            
            # Check if plugin checkpoint exists
            if [ ! -f "results/plugin_optimization/plugin_params.json" ]; then
                echo "❌ Plugin checkpoint not found!"
                exit 1
            fi
            
            # Check if experts exist
            if [ ! -d "checkpoints/balpoe_experts/cifar100_ir100" ]; then
                echo "❌ Experts directory not found!"
                exit 1
            fi
            
            # Run Stage 3
            python moe_plugin_pipeline/stage3_evaluate.py \
                --plugin_checkpoint results/plugin_optimization/plugin_params.json \
                --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
                --config moe_plugin_pipeline/configs/evaluation_config.json \
                --save_dir results/evaluation
            
            if [ $? -eq 0 ]; then
                echo "✅ Stage 3 completed successfully!"
            else
                echo "❌ Stage 3 failed!"
                exit 1
            fi
            ;;
        *)
            echo "❌ Unknown stage: $stage"
            exit 1
            ;;
    esac
done

echo ""
echo "🎉 MoE-Plugin Pipeline completed successfully!"
echo "📁 Results saved to:"
echo "  - Stage 1: checkpoints/balpoe_experts/cifar100_ir100/"
echo "  - Stage 2: results/plugin_optimization/"
echo "  - Stage 3: results/evaluation/"
echo ""
echo "📊 Ready for analysis and comparison!"
