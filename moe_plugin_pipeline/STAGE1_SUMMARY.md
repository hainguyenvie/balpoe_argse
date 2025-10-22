# 📋 Stage 1 Summary: BalPoE Experts Training

## ✅ Đã hoàn thành

### 1. **Config File** (`configs/cifar100_ir100_balpoe.json`)
- ✅ **Dataset**: CIFAR-100-LT với IR=100 (imb_factor=0.01)
- ✅ **Architecture**: ResNet-32 với 3 experts
- ✅ **Optimizer**: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- ✅ **LR Scheduler**: CustomLR (step1=160, step2=180, gamma=0.1)
- ✅ **Epochs**: 200 (Standard training)
- ✅ **Batch Size**: 128
- ✅ **Experts**: tau_list=[0, 1.0, 2.0] (λ ∈ {1, 0, -1})
- ✅ **Mixup**: alpha=0.4 (calibration bắt buộc)
- ✅ **Add extra info**: true (để lấy expert predictions)

### 2. **Training Script** (`stage1_train_balpoe.py`)
- ✅ **Sử dụng hoàn toàn code BalPoE gốc**
- ✅ **Không tạo hàm mới** - chỉ import và sử dụng
- ✅ **Tương thích với train.py gốc**
- ✅ **Support tất cả CLI options của BalPoE**

### 3. **Helper Scripts**
- ✅ **run_stage1.sh**: Script chạy đơn giản
- ✅ **test_stage1.py**: Kiểm tra setup
- ✅ **STAGE1_GUIDE.md**: Hướng dẫn chi tiết

## 🔍 Kiểm tra với yêu cầu của bạn

### ✅ **1.1 Environment & Data**
- ✅ Dataset: CIFAR-100-LT, IR=100
- ✅ imb_factor: 0.01 (tương đương IR=100)
- ✅ Architecture: ResNet-32

### ✅ **1.2 Training Hyperparameters**
- ✅ Optimizer: SGD với lr=0.1, momentum=0.9, weight_decay=5e-4
- ✅ LR Scheduler: Multi-step (160, 180 epochs, gamma=0.1)
- ✅ Epochs: 200
- ✅ Batch Size: 128

### ✅ **1.3 Expert Configuration**
- ✅ Số experts: 3
- ✅ Tau values: [0, 1.0, 2.0] (tương đương λ ∈ {1, 0, -1})
- ✅ Expert 1 (τ=0, λ=1): Head expert - Cross-Entropy
- ✅ Expert 2 (τ=1, λ=0): Balanced expert - Balanced Softmax
- ✅ Expert 3 (τ=2, λ=-1): Tail expert - Inverse distribution

### ✅ **1.4 Calibration**
- ✅ Method: Mixup
- ✅ Alpha: 0.4 (tối ưu cho CIFAR-100-LT)
- ✅ Target mix strategy: mix_input

## 🚀 Cách sử dụng

### **Test setup trước:**
```bash
python moe_plugin_pipeline/test_stage1.py
```

### **Chạy Stage 1:**
```bash
# Phương pháp 1: Script
./moe_plugin_pipeline/run_stage1.sh

# Phương pháp 2: Trực tiếp
python moe_plugin_pipeline/stage1_train_balpoe.py \
    -c moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json \
    -s 1

# Phương pháp 3: Sử dụng train.py gốc
python train.py -c moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json -s 1
```

## 📊 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:
```
checkpoints/balpoe_experts/cifar100_ir100/
├── config.json
├── checkpoint-epoch50.pth
├── checkpoint-epoch100.pth
├── checkpoint-epoch150.pth
├── checkpoint-epoch200.pth
└── model_best.pth
```

## 🔧 Không cần thêm hàm mới

Tất cả các hàm cần thiết đã có sẵn trong BalPoE gốc:
- ✅ `parse_tau_list()` - utils/util.py:282
- ✅ `learning_rate_scheduler()` - utils/util.py:331
- ✅ `write_json()` - utils/util.py:76
- ✅ `seed_everything()` - utils/util.py:322
- ✅ `ConfigParser` - parse_config.py:11
- ✅ `Trainer` - trainer/trainer.py
- ✅ `BSExpertLoss` - model/loss.py:8
- ✅ `Combiner` - utils/combiner.py:5

## ⚠️ Lưu ý quan trọng

1. **Chạy từ thư mục BalPoE gốc** - không phải từ moe_plugin_pipeline
2. **Calibration bắt buộc** - Mixup với α=0.4 là yêu cầu lý thuyết
3. **Expert diversity** - 3 experts tạo sự đa dạng tối đa
4. **Standard training** - 200 epochs để so sánh công bằng

## 🎯 Tiếp theo

Sau khi Stage 1 hoàn thành:
```bash
python moe_plugin_pipeline/stage2_optimize_plugin.py \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --config moe_plugin_pipeline/configs/plugin_optimization.json
```
