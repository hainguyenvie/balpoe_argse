# 🔧 Stage 2 Improvements

## 📊 **Phân tích vấn đề từ log gốc:**

### ❌ **Vấn đề chính:**
1. **Tail Expert bị bỏ qua hoàn toàn (0.00%)** - Mất đi sự đa dạng của ensemble
2. **Worst-group Error quá cao (90.39%)** - Performance trên tail classes kém
3. **Group weights mất cân bằng nghiêm trọng** - Algorithm tập trung vào head classes
4. **Không có kiểm tra chất lượng expert** - Không biết tại sao tail expert bị ignore

## 🔧 **Các cải thiện đã thêm:**

### **1. Expert Quality Analysis (`debug_expert_quality.py`)**
```python
# Phân tích chất lượng từng expert riêng lẻ
- Overall accuracy
- Head vs Tail accuracy  
- Confidence analysis
- Prediction diversity (entropy)
- Class bias analysis
```

**Lợi ích:**
- Hiểu tại sao tail expert bị bỏ qua
- Kiểm tra xem experts có được train đúng không
- Phát hiện bias trong predictions

### **2. Improved CS-plugin Optimization (`improved_optimization.py`)**

#### **A. Constrained Weight Search:**
```python
# Thêm constraints để đảm bảo tail expert được sử dụng
min_tail_expert_weight = 0.1      # Minimum 10% weight
min_balanced_expert_weight = 0.2  # Minimum 20% weight
```

#### **B. Weight Regularization:**
```python
# Penalty cho extreme weight distributions
- Penalty nếu expert nào có weight > 80%
- Penalty nếu tail expert có weight < 5%
```

#### **C. Better Alpha Initialization:**
```python
# Improved alpha finding dựa trên expert quality
- Better initialization
- Adaptive update rule
- Convergence checking
```

### **3. Improved Worst-group Optimization**

#### **A. Early Stopping:**
```python
# Tránh overfitting và convergence
patience = 5
no_improvement_count = 0
if no_improvement_count >= patience:
    break
```

#### **B. Convergence Detection:**
```python
# Dừng khi weights không thay đổi đáng kể
weight_change = torch.abs(group_weights - old_group_weights).sum().item()
if weight_change < 1e-6:
    break
```

#### **C. Stratified Data Splitting:**
```python
# Đảm bảo cả head và tail classes trong validation
val_head_size = min(val_size // 2, len(head_indices))
val_tail_size = val_size - val_head_size
```

## 🚀 **Cách sử dụng:**

### **Option 1: Chạy từng bước riêng lẻ**
```bash
# 1. Phân tích chất lượng expert
python moe_plugin_pipeline/debug_expert_quality.py \
    --experts_dir "checkpoints/balpoe_experts/cifar100_ir100/models/Imbalance_CIFAR100LT_IR100_BalPoE_Experts/1022_153647" \
    --config "moe_plugin_pipeline/configs/plugin_optimization.json"

# 2. Chạy improved optimization
python moe_plugin_pipeline/improved_optimization.py \
    --config "moe_plugin_pipeline/configs/plugin_optimization.json"
```

### **Option 2: Chạy tất cả cùng lúc**
```bash
python moe_plugin_pipeline/run_improved_stage2.py
```

## 📊 **Kết quả mong đợi:**

### **Trước (Original):**
- Expert weights: [0.429, 0.571, 0.000] ❌
- Tail expert: 0.00% (bị bỏ qua)
- Worst-group error: 90.39% (quá cao)

### **Sau (Improved):**
- Expert weights: [~0.3, ~0.4, ~0.3] ✅
- Tail expert: ~30% (được sử dụng)
- Worst-group error: <80% (cải thiện)
- Balanced error: <60% (cải thiện)

## 🔍 **Debug Information:**

### **Expert Quality Analysis sẽ cho biết:**
```
📊 head_expert Performance:
  - Overall Accuracy: 0.6234 (62.34%)
  - Head Accuracy: 0.6789 (67.89%)
  - Tail Accuracy: 0.4567 (45.67%)

📊 tail_expert Performance:
  - Overall Accuracy: 0.5890 (58.90%)
  - Head Accuracy: 0.5432 (54.32%)
  - Tail Accuracy: 0.6789 (67.89%)  ← Tốt hơn trên tail classes!
```

### **Improved Optimization sẽ cho biết:**
```
🔍 Constrained grid search: 378 weight combinations × 3 lambda values
✅ New best: balanced_error = 0.5890
📊 Weights: [0.300, 0.400, 0.300]  ← Tail expert được sử dụng!
```

## 💡 **Lợi ích chính:**

1. **Tận dụng đầy đủ ensemble diversity** - Tất cả experts được sử dụng
2. **Cải thiện performance trên tail classes** - Worst-group error giảm
3. **Tránh overfitting** - Early stopping và regularization
4. **Debug-friendly** - Hiểu rõ tại sao có kết quả như vậy
5. **Robust optimization** - Constraints đảm bảo kết quả hợp lý

## 🎯 **Next Steps:**

Sau khi chạy improved optimization, bạn có thể:
1. So sánh kết quả với original optimization
2. Chạy Stage 3 với improved parameters
3. Phân tích Risk-Coverage curves để thấy sự cải thiện
4. Tune thêm parameters nếu cần
