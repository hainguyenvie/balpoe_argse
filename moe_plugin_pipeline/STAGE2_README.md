# 🎯 Stage 2: Tối ưu hóa Plugin Parameters

## 📋 Tổng quan

Giai đoạn 2 sử dụng **3 bộ xác suất hậu nghiệm từ các chuyên gia** (Head, Balanced, Tail) để tìm ra một **"hỗn hợp" tối ưu** và các tham số plugin `(α̂, μ̂)` nhằm tối thiểu hóa balanced error và worst-group error.

## ⚙️ Cấu hình Chi tiết

### 2.1 Chuẩn bị Môi trường và Dữ liệu

#### 2.1.1 Phân chia Dữ liệu
- **Validation Set (20%)**: Tìm kiếm siêu tham số tối ưu
- **Test Set (80%)**: Đánh giá cuối cùng trong Giai đoạn 3
- **Lý do**: Ngăn chặn rò rỉ thông tin từ tập test

#### 2.1.2 Định nghĩa Nhóm (Head vs. Tail)
- **Nhóm Tail**: Các lớp có ≤ 20 mẫu trong training set
- **Nhóm Head**: Các lớp còn lại
- **Lý do**: Định nghĩa chính xác theo paper "Learning to Reject"

#### 2.1.3 Chuẩn bị Đầu vào cho Plugin
- **Input**: 3 vector xác suất hậu nghiệm từ experts
- **Output**: Hỗn hợp có trọng số `p_mix(y|x) = w_head·p_head(y|x) + w_bal·p_bal(y|x) + w_tail·p_tail(y|x)`

### 2.2 Thuật toán 1: CS-plugin để tối ưu Balanced Error

#### 2.2.1 Grid Search cho Expert Weights
- **Search space**: `w_head, w_bal, w_tail ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`
- **Constraint**: Tổng weights = 1
- **Purpose**: Tìm trọng số hỗn hợp tối ưu

#### 2.2.2 Grid Search cho λ₀
- **Search space**: `λ₀ ∈ {1, 6, 11}` (theo Phụ lục F.3)
- **Purpose**: Tìm tham số plugin tối ưu

#### 2.2.3 Power Iteration cho α
- **Iterations**: M = 10 (theo Phụ lục F.3)
- **Purpose**: Tìm α tối ưu cho mỗi cặp (w, λ₀)

### 2.3 Thuật toán 2: Worst-group Plugin để tối ưu Worst-Group Error

#### 2.3.1 Cấu hình Thuật toán
- **Số vòng lặp**: T = 25
- **Step-size**: ξ = 1.0
- **Khởi tạo**: β⁽⁰⁾ = {0.5, 0.5}

#### 2.3.2 Quy trình Triển khai
1. **Vòng lặp t = 0 đến T-1**:
   - Gọi CS-plugin với trọng số nhóm β⁽ᵗ⁾
   - Tính group-wise errors
   - Cập nhật β⁽ᵗ⁺¹⁾ bằng exponentiated gradient

2. **Cập nhật trọng số nhóm**:
   ```
   βₖ⁽ᵗ⁺¹⁾ ∝ βₖ⁽ᵗ⁾ · exp(ξ · êₖ(h⁽ᵗ⁾, r⁽ᵗ⁾))
   ```

## 🚀 Cách chạy

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

## 🔍 Kiểm tra kết quả

### 1. Kiểm tra optimized parameters
```bash
cat checkpoints/plugin_optimized/optimized_parameters.json
```

### 2. Kiểm tra log optimization
- Logs sẽ hiển thị trong terminal
- Hiển thị progress của grid search
- Hiển thị best parameters tại mỗi bước

### 3. Kiểm tra config đã sử dụng
```bash
cat checkpoints/plugin_optimized/config.json
```

## ⚠️ Lưu ý quan trọng

1. **Sử dụng hoàn toàn code BalPoE gốc** - không tạo hàm mới
2. **Tuân thủ đúng thuật toán từ paper** - grid search chính xác
3. **Validation set (20%)** - không sử dụng test set
4. **Group definition** - Tail = classes với ≤ 20 samples

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **"No checkpoint files found"**
   - Chạy Stage 1 trước để tạo expert checkpoints
   - Kiểm tra đường dẫn experts_dir

2. **"CUDA out of memory"**
   - Giảm batch_size trong config
   - Sử dụng CPU thay vì GPU

3. **"Dataset not found"**
   - Đảm bảo có thư mục ./data/CIFAR-100
   - Dataset sẽ được tự động tạo nếu chưa có

## 📈 Monitoring

Trong quá trình optimization, bạn sẽ thấy:
- **CS-plugin progress**: Grid search cho expert weights và λ₀
- **Worst-group progress**: Vòng lặp t = 0 đến 24
- **Best parameters**: Cập nhật khi tìm được kết quả tốt hơn
- **Final results**: Balanced error và worst-group error

## 🔄 Tiếp theo

Sau khi Stage 2 hoàn thành:
```bash
python moe_plugin_pipeline/stage3_evaluate.py \
    --plugin_checkpoint checkpoints/plugin_optimized/optimized_parameters.json \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --config moe_plugin_pipeline/configs/plugin_optimization.json \
    --save_dir results/evaluation \
    --seed 1
```

## 📚 Tài liệu tham khảo

- **Paper gốc**: "Learning to Reject Meets Long-tail Learning"
- **BalPoE paper**: "Balanced Product of Calibrated Experts for Long-Tailed Recognition"
- **Config files**: `moe_plugin_pipeline/configs/plugin_optimization.json`
- **Test script**: `moe_plugin_pipeline/test_stage2.py`
