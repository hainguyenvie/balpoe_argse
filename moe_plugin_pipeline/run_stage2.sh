#!/bin/bash
# Script để chạy Stage 2: Optimize Plugin Parameters
# Sử dụng hoàn toàn code từ BalPoE gốc

echo "🚀 BẮT ĐẦU STAGE 2: TỐI ƯU PLUGIN PARAMETERS"
echo "=============================================="

# Kiểm tra xem có trong thư mục BalPoE gốc không
if [ ! -f "train.py" ]; then
    echo "❌ Không tìm thấy train.py. Vui lòng chạy từ thư mục BalPoE gốc."
    exit 1
fi

# Kiểm tra xem có experts checkpoints không
if [ ! -d "checkpoints/balpoe_experts/cifar100_ir100" ]; then
    echo "❌ Không tìm thấy experts checkpoints."
    echo "💡 Vui lòng chạy Stage 1 trước:"
    echo "   python moe_plugin_pipeline/stage1_train_balpoe.py \\"
    echo "       -c moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json \\"
    echo "       -s 1"
    exit 1
fi

# Chạy stage 2 với config đã chuẩn bị
python moe_plugin_pipeline/stage2_optimize_plugin.py \
    --config moe_plugin_pipeline/configs/plugin_optimization.json \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --seed 1

echo "✅ Stage 2 hoàn thành!"
echo "📁 Optimized parameters: checkpoints/plugin_optimized/optimized_parameters.json"
echo "🔧 Tiếp theo: python moe_plugin_pipeline/stage3_evaluate.py --plugin_checkpoint checkpoints/plugin_optimized/optimized_parameters.json"
