#!/bin/bash
# Script để chạy Stage 1: Train BalPoE Experts
# Sử dụng hoàn toàn code từ BalPoE gốc

echo "🚀 BẮT ĐẦU STAGE 1: HUẤN LUYỆN BALPOE EXPERTS"
echo "=============================================="

# Kiểm tra xem có trong thư mục BalPoE gốc không
if [ ! -f "train.py" ]; then
    echo "❌ Không tìm thấy train.py. Vui lòng chạy từ thư mục BalPoE gốc."
    exit 1
fi

# Chạy stage 1 với config đã chuẩn bị
python moe_plugin_pipeline/stage1_train_balpoe.py \
    -c moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json \
    -s 1

echo "✅ Stage 1 hoàn thành!"
echo "📁 Checkpoints saved to: checkpoints/balpoe_experts/cifar100_ir100"
echo "🔧 Tiếp theo: python moe_plugin_pipeline/stage2_optimize_plugin.py --experts_dir checkpoints/balpoe_experts/cifar100_ir100"
