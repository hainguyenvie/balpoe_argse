# Stage 3: Evaluation và Comparison

## Mục tiêu
Đánh giá hiệu suất của MoE-Plugin và so sánh với các baselines theo đúng protocol từ paper "Learning to Reject Meets Long-tail Learning".

## Cấu trúc Stage 3

### 1. Files chính
- `stage3_evaluate.py`: Script chính cho evaluation
- `configs/evaluation_config.json`: Config cho evaluation
- `test_stage3.py`: Test script
- `run_stage3.sh`: Script chạy Stage 3

### 2. Evaluation Protocol

#### 2.1. Risk-Coverage Curves
- **Mục tiêu**: Tạo đường cong trade-off giữa error và rejection rate
- **Cost values**: Từ 0.0 đến 1.0 với bước nhảy 0.05 (21 điểm)
- **Metrics**: Balanced Error và Worst-Group Error vs Rejection Rate

#### 2.2. AURC (Area Under Risk-Coverage Curve)
- **Mục tiêu**: Tóm tắt hiệu suất tổng thể
- **Metrics**: AURC cho Balanced Error và Worst-Group Error
- **Công thức**: `AURC = ∫ error(rejection_rate) d(rejection_rate)`

#### 2.3. Baseline Methods
1. **BalPoE (gốc)**: Average ensemble, không có rejection
2. **Plugin_Single**: Plugin trên single balanced expert
3. **Plugin_BalPoE_avg**: Plugin trên averaged BalPoE
4. **MoE_Plugin**: Proposed method với learned weights

### 3. Cấu hình

#### 3.1. Config file (`configs/evaluation_config.json`)
```json
{
    "name": "MoE-Plugin_Evaluation",
    "dataset": {
        "name": "CIFAR-100-LT",
        "data_dir": "./data/CIFAR-100",
        "imbalance_ratio": 100
    },
    "group_definition": {
        "tail_threshold": 20,
        "head_threshold": 100
    },
    "evaluation": {
        "cost_values": [0.0, 0.05, 0.1, ..., 1.0],
        "metrics": ["balanced_error", "worst_group_error", "rejection_rate"],
        "baselines": ["BalPoE", "Plugin_Single", "Plugin_BalPoE_avg", "MoE_Plugin"]
    },
    "output": {
        "save_dir": "results/evaluation",
        "plots": true,
        "tables": true,
        "detailed_results": true
    }
}
```

#### 3.2. Required inputs
- **Plugin checkpoint**: File JSON chứa optimized parameters từ Stage 2
- **Experts directory**: Thư mục chứa expert checkpoints từ Stage 1
- **Config file**: Evaluation configuration

### 4. Cách chạy

#### 4.1. Test setup
```bash
cd moe_plugin_pipeline
python test_stage3.py
```

#### 4.2. Chạy Stage 3
```bash
# Sử dụng script
./run_stage3.sh <plugin_checkpoint> <experts_dir> <config_file> [save_dir] [seed]

# Hoặc chạy trực tiếp
python stage3_evaluate.py \
    --plugin_checkpoint results/plugin_optimization/plugin_params.json \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --config configs/evaluation_config.json \
    --save_dir results/evaluation \
    --seed 1
```

### 5. Output Files

#### 5.1. Risk-Coverage Curves
- **File**: `risk_coverage_curves.png`
- **Nội dung**: 2 đồ thị
  - Balanced Error vs Rejection Rate
  - Worst-Group Error vs Rejection Rate
- **Lines**: 4 methods (BalPoE, Plugin_Single, Plugin_BalPoE_avg, MoE_Plugin)

#### 5.2. AURC Comparison Table
- **File**: `aurc_comparison.json`
- **Nội dung**: AURC values cho mỗi method
- **Format**:
```json
[
    {
        "Method": "BalPoE",
        "Balanced AURC": "0.1234",
        "Worst-Group AURC": "0.2345"
    },
    ...
]
```

#### 5.3. Accuracy Comparison Table
- **File**: `accuracy_comparison.json`
- **Nội dung**: Accuracy tại 0% rejection
- **Format**:
```json
[
    {
        "Method": "BalPoE",
        "Balanced Error": "0.1234",
        "Worst-Group Error": "0.2345",
        "Rejection Rate": "0.0000"
    },
    ...
]
```

### 6. Evaluation Logic

#### 6.1. MoEPluginEvaluator Class
- **Mục tiêu**: Orchestrate toàn bộ evaluation process
- **Methods**:
  - `setup_data_loaders()`: Setup test data loader
  - `load_expert_models()`: Load 3 expert models
  - `get_expert_predictions()`: Get predictions từ experts
  - `define_groups()`: Define Head vs Tail groups
  - `create_risk_coverage_curves()`: Create curves cho tất cả methods
  - `compute_aurc()`: Compute AURC values
  - `plot_risk_coverage_curves()`: Create plots
  - `create_comparison_tables()`: Create comparison tables

#### 6.2. Baseline Evaluation Methods
- **`_evaluate_balpoe_baseline()`**: BalPoE gốc (average ensemble)
- **`_evaluate_plugin_single()`**: Plugin trên single expert
- **`_evaluate_plugin_balpoe_avg()`**: Plugin trên averaged BalPoE
- **`_evaluate_moe_plugin()`**: Proposed MoE-Plugin method

#### 6.3. Error Computation
- **`_compute_balanced_error()`**: Balanced error = (head_error + tail_error) / 2
- **`_compute_worst_group_error()`**: Worst-group error = max(head_error, tail_error)
- **`_compute_group_error()`**: Error cho một group cụ thể

### 7. Expected Results

#### 7.1. AURC Comparison
- **MoE-Plugin**: Lowest AURC (best performance)
- **Plugin_BalPoE_avg**: Second best
- **Plugin_Single**: Third
- **BalPoE**: Highest AURC (no rejection capability)

#### 7.2. Risk-Coverage Curves
- **MoE-Plugin**: Curve nằm dưới tất cả baselines
- **Plugin_BalPoE_avg**: Curve nằm dưới Plugin_Single
- **Plugin_Single**: Curve nằm dưới BalPoE
- **BalPoE**: Horizontal line (no rejection)

#### 7.3. Accuracy Analysis
- **MoE-Plugin**: Best accuracy trên Few-shot classes
- **BalPoE**: Good accuracy trên Head classes
- **Plugin methods**: Better trade-off giữa accuracy và rejection

### 8. Troubleshooting

#### 8.1. Common Issues
- **Missing plugin checkpoint**: Kiểm tra Stage 2 đã hoàn thành
- **Missing expert checkpoints**: Kiểm tra Stage 1 đã hoàn thành
- **Config errors**: Kiểm tra config file format
- **Memory issues**: Giảm batch size trong data loader

#### 8.2. Debug Tips
- Chạy `test_stage3.py` để kiểm tra setup
- Kiểm tra log output để debug
- Verify expert models được load đúng
- Check plugin parameters format

### 9. Next Steps

Sau khi Stage 3 hoàn thành:
1. **Analyze results**: Xem xét AURC values và curves
2. **Compare baselines**: So sánh performance của các methods
3. **Visualize results**: Tạo plots và tables cho paper
4. **Document findings**: Ghi lại kết quả và insights

## Kết luận

Stage 3 cung cấp một framework đánh giá toàn diện cho MoE-Plugin pipeline, đảm bảo so sánh công bằng với các baselines và tuân thủ đúng evaluation protocol từ paper "Learning to Reject Meets Long-tail Learning".
