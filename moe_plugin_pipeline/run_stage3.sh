#!/bin/bash

# Stage 3: Evaluation và Comparison
# Chạy đánh giá MoE-Plugin và so sánh với baselines

echo "🚀 Stage 3: Evaluation và Comparison"
echo "=================================="

# Kiểm tra arguments
if [ $# -lt 3 ]; then
    echo "❌ Usage: $0 <plugin_checkpoint> <experts_dir> <config_file> [save_dir] [seed]"
    echo "   Example: $0 results/plugin_optimization/plugin_params.json checkpoints/balpoe_experts/cifar100_ir100 configs/evaluation_config.json results/evaluation 1"
    exit 1
fi

PLUGIN_CHECKPOINT=$1
EXPERTS_DIR=$2
CONFIG_FILE=$3
SAVE_DIR=${4:-"results/evaluation"}
SEED=${5:-1}

echo "📂 Plugin checkpoint: $PLUGIN_CHECKPOINT"
echo "📂 Experts directory: $EXPERTS_DIR"
echo "📂 Config file: $CONFIG_FILE"
echo "📂 Save directory: $SAVE_DIR"
echo "🎲 Random seed: $SEED"

# Kiểm tra files tồn tại
if [ ! -f "$PLUGIN_CHECKPOINT" ]; then
    echo "❌ Plugin checkpoint not found: $PLUGIN_CHECKPOINT"
    exit 1
fi

if [ ! -d "$EXPERTS_DIR" ]; then
    echo "❌ Experts directory not found: $EXPERTS_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Tạo thư mục save_dir
mkdir -p "$SAVE_DIR"

echo "✅ All files and directories found"

# Chạy Stage 3
echo ""
echo "🔍 Running Stage 3: Evaluation..."
echo "=================================="

python moe_plugin_pipeline/stage3_evaluate.py \
    --plugin_checkpoint "$PLUGIN_CHECKPOINT" \
    --experts_dir "$EXPERTS_DIR" \
    --config "$CONFIG_FILE" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED"

# Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Stage 3 completed successfully!"
    echo "📁 Results saved to: $SAVE_DIR"
    echo ""
    echo "📊 Generated files:"
    echo "  - risk_coverage_curves.png: Risk-Coverage curves"
    echo "  - aurc_comparison.json: AURC comparison table"
    echo "  - accuracy_comparison.json: Accuracy comparison table"
    echo ""
    echo "🎉 MoE-Plugin Pipeline completed!"
    echo "📈 Ready for analysis and comparison!"
else
    echo ""
    echo "❌ Stage 3 failed!"
    echo "🔍 Check the error messages above"
    exit 1
fi
