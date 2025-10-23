#!/bin/bash

# Improved Stage 2: Analysis + Optimization
# This script runs expert quality analysis and improved optimization

echo "🚀 Running Improved Stage 2: Analysis + Optimization"
echo "============================================================"

# Check if config exists
CONFIG_FILE="moe_plugin_pipeline/configs/plugin_optimization.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Step 1: Analyze expert quality
echo "📊 Step 1: Analyzing Expert Quality..."
echo "----------------------------------------"
python moe_plugin_pipeline/debug_expert_quality.py \
    --experts_dir "checkpoints/balpoe_experts/cifar100_ir100/models/Imbalance_CIFAR100LT_IR100_BalPoE_Experts/1022_153647" \
    --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Expert quality analysis failed"
    exit 1
fi

echo ""
echo "📊 Step 2: Running Improved Optimization..."
echo "----------------------------------------"
python moe_plugin_pipeline/improved_optimization.py \
    --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Improved optimization failed"
    exit 1
fi

echo ""
echo "✅ Improved Stage 2 completed!"
echo "📁 Results saved to: checkpoints/plugin_optimized/"
echo "🔧 Next step: python moe_plugin_pipeline/stage3_evaluate.py --plugin_checkpoint checkpoints/plugin_optimized/improved_optimized_parameters.json"
