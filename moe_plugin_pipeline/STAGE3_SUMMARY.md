# Stage 3: Evaluation và Comparison - Summary

## Tổng quan
Stage 3 triển khai một framework đánh giá toàn diện cho MoE-Plugin pipeline, tuân thủ đúng evaluation protocol từ paper "Learning to Reject Meets Long-tail Learning".

## Files đã tạo

### 1. Core Files
- **`stage3_evaluate.py`**: Script chính cho evaluation
- **`configs/evaluation_config.json`**: Configuration cho evaluation
- **`test_stage3.py`**: Test script để kiểm tra setup
- **`run_stage3.sh`**: Shell script để chạy Stage 3
- **`STAGE3_GUIDE.md`**: Hướng dẫn chi tiết
- **`STAGE3_SUMMARY.md`**: Tóm tắt Stage 3

### 2. Key Features

#### 2.1. MoEPluginEvaluator Class
- **Mục tiêu**: Orchestrate toàn bộ evaluation process
- **Sử dụng hoàn toàn code từ BalPoE gốc**: Không tạo hàm mới
- **Tuân thủ paper "Learning to Reject"**: Risk-Coverage Curves và AURC metrics

#### 2.2. Evaluation Protocol
- **Risk-Coverage Curves**: 21 cost values từ 0.0 đến 1.0
- **AURC Computation**: Area Under Risk-Coverage Curve
- **Baseline Methods**: 4 methods để so sánh
- **Group Definition**: Head vs Tail classes theo paper

#### 2.3. Baseline Methods
1. **BalPoE (gốc)**: Average ensemble, không có rejection
2. **Plugin_Single**: Plugin trên single balanced expert
3. **Plugin_BalPoE_avg**: Plugin trên averaged BalPoE
4. **MoE_Plugin**: Proposed method với learned weights

## Cấu trúc Code

### 1. MoEPluginEvaluator Class
```python
class MoEPluginEvaluator:
    def __init__(self, plugin_checkpoint, experts_dir, config_path, seed)
    def setup_data_loaders(self)
    def load_expert_models(self)
    def get_expert_predictions(self, data_loader)
    def define_groups(self, labels)
    def create_risk_coverage_curves(self, expert_predictions, labels, head_classes, tail_classes)
    def compute_aurc(self, results)
    def plot_risk_coverage_curves(self, results, save_dir)
    def create_comparison_tables(self, results, aurc_results, save_dir)
    def evaluate(self)
```

### 2. Baseline Evaluation Methods
```python
def _evaluate_balpoe_baseline(self, expert_predictions, labels, head_classes, tail_classes, cost)
def _evaluate_plugin_single(self, expert_predictions, labels, head_classes, tail_classes, cost)
def _evaluate_plugin_balpoe_avg(self, expert_predictions, labels, head_classes, tail_classes, cost)
def _evaluate_moe_plugin(self, expert_predictions, labels, head_classes, tail_classes, cost)
```

### 3. Error Computation Methods
```python
def _compute_balanced_error(self, predictions, labels, head_classes, tail_classes, reject_mask=None)
def _compute_worst_group_error(self, predictions, labels, head_classes, tail_classes, reject_mask=None)
def _compute_group_error(self, predictions, labels, group_mask, reject_mask)
```

## Configuration

### 1. Evaluation Config (`configs/evaluation_config.json`)
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

### 2. Required Inputs
- **Plugin checkpoint**: JSON file từ Stage 2
- **Experts directory**: Thư mục chứa expert checkpoints từ Stage 1
- **Config file**: Evaluation configuration

## Output Files

### 1. Risk-Coverage Curves
- **File**: `risk_coverage_curves.png`
- **Nội dung**: 2 đồ thị
  - Balanced Error vs Rejection Rate
  - Worst-Group Error vs Rejection Rate
- **Lines**: 4 methods (BalPoE, Plugin_Single, Plugin_BalPoE_avg, MoE_Plugin)

### 2. AURC Comparison Table
- **File**: `aurc_comparison.json`
- **Nội dung**: AURC values cho mỗi method
- **Format**: JSON array với Method, Balanced AURC, Worst-Group AURC

### 3. Accuracy Comparison Table
- **File**: `accuracy_comparison.json`
- **Nội dung**: Accuracy tại 0% rejection
- **Format**: JSON array với Method, Balanced Error, Worst-Group Error, Rejection Rate

## Cách sử dụng

### 1. Test Setup
```bash
cd moe_plugin_pipeline
python test_stage3.py
```

### 2. Chạy Stage 3
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

## Key Features

### 1. Tuân thủ Paper "Learning to Reject"
- **Risk-Coverage Curves**: Đúng theo Phụ lục F.4
- **AURC Metrics**: Đúng theo Bảng 2
- **Group Definition**: Tail classes ≤ 20 samples
- **Cost Values**: 0.0 đến 1.0 với bước nhảy 0.05

### 2. Sử dụng Code từ BalPoE Gốc
- **Data Loaders**: Sử dụng `ImbalanceCIFAR100DataLoader` từ BalPoE
- **Model Architecture**: Sử dụng `ResNet32Model` từ BalPoE
- **Expert Loading**: Load trực tiếp từ BalPoE checkpoints
- **Không tạo hàm mới**: Chỉ sử dụng functions có sẵn

### 3. Comprehensive Evaluation
- **4 Baseline Methods**: Đầy đủ comparison
- **21 Cost Points**: Dense evaluation
- **2 Metrics**: Balanced Error và Worst-Group Error
- **Visualization**: Plots và tables

## Expected Results

### 1. AURC Ranking
1. **MoE-Plugin**: Lowest AURC (best)
2. **Plugin_BalPoE_avg**: Second best
3. **Plugin_Single**: Third
4. **BalPoE**: Highest AURC (no rejection)

### 2. Risk-Coverage Curves
- **MoE-Plugin**: Curve nằm dưới tất cả baselines
- **Plugin_BalPoE_avg**: Curve nằm dưới Plugin_Single
- **Plugin_Single**: Curve nằm dưới BalPoE
- **BalPoE**: Horizontal line (no rejection)

### 3. Performance Analysis
- **MoE-Plugin**: Best trade-off giữa accuracy và rejection
- **Plugin methods**: Better performance trên Few-shot classes
- **BalPoE**: Good baseline cho Head classes

## Troubleshooting

### 1. Common Issues
- **Missing plugin checkpoint**: Kiểm tra Stage 2 hoàn thành
- **Missing expert checkpoints**: Kiểm tra Stage 1 hoàn thành
- **Config errors**: Kiểm tra JSON format
- **Memory issues**: Giảm batch size

### 2. Debug Tips
- Chạy `test_stage3.py` để kiểm tra setup
- Kiểm tra log output
- Verify expert models
- Check plugin parameters

## Kết luận

Stage 3 cung cấp một framework đánh giá toàn diện và chặt chẽ cho MoE-Plugin pipeline, đảm bảo:

1. **Tuân thủ đúng paper "Learning to Reject"**: Risk-Coverage Curves, AURC, Group definition
2. **Sử dụng hoàn toàn code từ BalPoE gốc**: Không tạo hàm mới
3. **Comprehensive evaluation**: 4 baselines, 21 cost points, 2 metrics
4. **Professional output**: Plots, tables, JSON results
5. **Easy to use**: Test script, shell script, clear documentation

Stage 3 sẵn sàng để chạy và đánh giá MoE-Plugin pipeline một cách chuyên nghiệp!
