# 📋 Stage 2 Summary: Plugin Optimization

## ✅ Đã hoàn thành

### 1. **Config File** (`configs/plugin_optimization.json`)
- ✅ **Dataset**: CIFAR-100-LT với validation split 20%
- ✅ **Group Definition**: Tail = classes với ≤ 20 samples
- ✅ **CS-plugin**: λ₀ ∈ {1, 6, 11}, M=10 iterations
- ✅ **Expert Weights**: Grid search {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- ✅ **Worst-group Plugin**: T=25 iterations, ξ=1.0
- ✅ **Output**: Save optimized parameters

### 2. **Optimization Script** (`stage2_optimize_plugin.py`)
- ✅ **Sử dụng hoàn toàn code BalPoE gốc**
- ✅ **Không tạo hàm mới** - chỉ import và sử dụng
- ✅ **Tuân thủ đúng thuật toán từ paper**
- ✅ **Grid search chính xác theo yêu cầu**

### 3. **Helper Scripts**
- ✅ **run_stage2.sh**: Script chạy đơn giản
- ✅ **test_stage2.py**: Kiểm tra setup
- ✅ **STAGE2_GUIDE.md**: Hướng dẫn chi tiết

## 🔍 Kiểm tra với yêu cầu của bạn

### ✅ **2.1 Chuẩn bị Môi trường và Dữ liệu**
- ✅ **Validation Set (20%)**: Tìm kiếm siêu tham số tối ưu
- ✅ **Test Set (80%)**: Đánh giá cuối cùng
- ✅ **Group Definition**: Tail = classes với ≤ 20 samples
- ✅ **Expert Predictions**: 3 vector xác suất hậu nghiệm

### ✅ **2.2 Thuật toán 1: CS-plugin**
- ✅ **Grid Search cho Expert Weights**: {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- ✅ **Grid Search cho λ₀**: {1, 6, 11} (theo Phụ lục F.3)
- ✅ **Power Iteration cho α**: M=10 iterations
- ✅ **Balanced Error Optimization**: Tìm (w*, λ₀*)

### ✅ **2.3 Thuật toán 2: Worst-group Plugin**
- ✅ **Số vòng lặp**: T=25
- ✅ **Step-size**: ξ=1.0
- ✅ **Khởi tạo**: β⁽⁰⁾ = {0.5, 0.5}
- ✅ **Exponentiated Gradient**: Cập nhật β⁽ᵗ⁺¹⁾
- ✅ **Worst-group Error Optimization**: Tìm (h⁽ᵀ⁾, r⁽ᵀ⁾)

## 🚀 Cách sử dụng

### **Test setup trước:**
```bash
python moe_plugin_pipeline/test_stage2.py
```

### **Chạy Stage 2:**
```bash
# Phương pháp 1: Script
./moe_plugin_pipeline/run_stage2.sh

# Phương pháp 2: Trực tiếp
python moe_plugin_pipeline/stage2_optimize_plugin.py \
    --config moe_plugin_pipeline/configs/plugin_optimization.json \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --seed 1
```

## 📊 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:
```
checkpoints/plugin_optimized/
├── optimized_parameters.json    # Các tham số đã tối ưu
└── config.json                 # Config đã sử dụng
```

### **Nội dung optimized_parameters.json:**
```json
{
  "lambda_0": 6,
  "alpha": 0.7,
  "expert_weights": [0.3, 0.4, 0.3],
  "group_weights": [0.6, 0.4],
  "rejection_threshold": 0.7,
  "balanced_error": 0.25,
  "worst_group_error": 0.30
}
```

## 🔧 Không cần thêm hàm mới

Tất cả các hàm cần thiết đã có sẵn trong BalPoE gốc:
- ✅ `seed_everything()` - utils/util.py:322
- ✅ `ConfigParser` - parse_config.py:11
- ✅ `ImbalanceCIFAR100DataLoader` - data_loader/data_loaders.py
- ✅ `ResNet32Model` - model/model.py
- ✅ `torch.softmax()`, `torch.max()` - PyTorch built-in
- ✅ `torch.utils.data.DataLoader` - PyTorch built-in

## ⚠️ Lưu ý quan trọng

1. **Chạy từ thư mục BalPoE gốc** - không phải từ moe_plugin_pipeline
2. **Tuân thủ đúng thuật toán từ paper** - grid search chính xác
3. **Validation set (20%)** - không sử dụng test set
4. **Group definition** - Tail = classes với ≤ 20 samples

## 🎯 Tiếp theo

Sau khi Stage 2 hoàn thành:
```bash
python moe_plugin_pipeline/stage3_evaluate.py \
    --plugin_checkpoint checkpoints/plugin_optimized/optimized_parameters.json \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --config moe_plugin_pipeline/configs/plugin_optimization.json \
    --save_dir results/evaluation \
    --seed 1
```
