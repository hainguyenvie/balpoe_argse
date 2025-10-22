# 🎯 Giai đoạn 1: Huấn luyện BalPoE Experts - Hướng dẫn Chi tiết

## 📋 Tổng quan

Giai đoạn 1 tạo ra **3 chuyên gia ResNet-32 đa dạng và được hiệu chỉnh tốt** làm đầu vào cho các thuật toán plugin. Sử dụng hoàn toàn code từ BalPoE gốc, không tạo hàm mới.

## ⚙️ Cấu hình Chi tiết

### 1.1 Dataset & Environment
- **Dataset**: CIFAR-100-LT với IR=100 (imb_factor=0.01)
- **Architecture**: ResNet-32 với 3 experts
- **Device**: CUDA (nếu có) hoặc CPU

### 1.2 Training Hyperparameters (Standard Training)
- **Optimizer**: SGD
  - Learning Rate: 0.1
  - Momentum: 0.9
  - Weight Decay: 5e-4
  - Nesterov: true
- **LR Scheduler**: CustomLR (Multi-step)
  - Step 1: 160 epochs
  - Step 2: 180 epochs
  - Gamma: 0.1
  - Warmup: 5 epochs
- **Epochs**: 200
- **Batch Size**: 128

### 1.3 Expert Configuration (Trái tim của BalPoE)
- **Số experts**: 3
- **Tau values**: [0, 1.0, 2.0] (tương đương λ ∈ {1, 0, -1})
  - **Expert 1 (τ=0, λ=1)**: Head expert - Cross-Entropy standard
  - **Expert 2 (τ=1, λ=0)**: Balanced expert - Balanced Softmax  
  - **Expert 3 (τ=2, λ=-1)**: Tail expert - Inverse distribution

### 1.4 Calibration (Yêu cầu bắt buộc)
- **Method**: Mixup
- **Alpha**: 0.4 (tối ưu cho CIFAR-100-LT)
- **Target mix strategy**: mix_input

## 🚀 Cách chạy

### Phương pháp 1: Sử dụng script
```bash
# Từ thư mục BalPoE gốc
chmod +x moe_plugin_pipeline/run_stage1.sh
./moe_plugin_pipeline/run_stage1.sh
```

### Phương pháp 2: Chạy trực tiếp
```bash
# Từ thư mục BalPoE gốc
python moe_plugin_pipeline/stage1_train_balpoe.py \
    -c moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json \
    -s 1
```

### Phương pháp 3: Sử dụng train.py gốc với config
```bash
# Từ thư mục BalPoE gốc
python train.py -c moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json -s 1
```

## 📊 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:

```
checkpoints/balpoe_experts/cifar100_ir100/
├── config.json                    # Config đã sử dụng
├── checkpoint-epoch50.pth         # Checkpoint tại epoch 50
├── checkpoint-epoch100.pth       # Checkpoint tại epoch 100
├── checkpoint-epoch150.pth       # Checkpoint tại epoch 150
├── checkpoint-epoch200.pth       # Checkpoint cuối cùng
└── model_best.pth               # Model tốt nhất (nếu có)
```

## 🔍 Kiểm tra kết quả

### 1. Kiểm tra config đã sử dụng
```bash
cat checkpoints/balpoe_experts/cifar100_ir100/config.json
```

### 2. Kiểm tra log training
- Logs sẽ hiển thị trong terminal
- Có thể sử dụng tensorboard nếu được cấu hình

### 3. Kiểm tra checkpoints
```python
import torch
checkpoint = torch.load('checkpoints/balpoe_experts/cifar100_ir100/checkpoint-epoch200.pth')
print("Keys:", checkpoint.keys())
print("Epoch:", checkpoint['epoch'])
print("Model state dict keys:", list(checkpoint['state_dict'].keys())[:5])
```

## ⚠️ Lưu ý quan trọng

1. **Không tạo hàm mới**: Stage 1 sử dụng hoàn toàn code từ BalPoE gốc
2. **Calibration bắt buộc**: Mixup với α=0.4 là yêu cầu lý thuyết
3. **Expert diversity**: 3 experts với τ ∈ {0, 1, 2} tạo sự đa dạng tối đa
4. **Standard training**: 200 epochs để so sánh công bằng với BalPoE gốc

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **"No module named 'utils'"**
   - Đảm bảo chạy từ thư mục BalPoE gốc
   - Kiểm tra sys.path trong script

2. **"CUDA out of memory"**
   - Giảm batch_size trong config
   - Sử dụng n_gpu: 1 thay vì nhiều GPU

3. **"Dataset not found"**
   - Đảm bảo có thư mục ./data/CIFAR-100
   - Dataset sẽ được tự động tạo nếu chưa có

## 📈 Monitoring

Trong quá trình training, bạn sẽ thấy:
- Training loss cho từng expert
- Validation accuracy
- Learning rate schedule
- Mixup alpha values

## 🔄 Tiếp theo

Sau khi Stage 1 hoàn thành:
```bash
python moe_plugin_pipeline/stage2_optimize_plugin.py \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --config moe_plugin_pipeline/configs/plugin_optimization.json
```
