# MoE-Plugin Pipeline

## Tổng quan
Pipeline kết hợp **BalPoE (Balanced Product of Calibrated Experts)** với **Plugin Rejection Method** từ paper "Learning to Reject Meets Long-tail Learning" để tạo ra một phương pháp mới: **MoE-Plugin**.

## Kiến trúc Pipeline

### Stage 1: Train BalPoE Experts
- **Mục tiêu**: Huấn luyện 3 expert models đa dạng và được hiệu chỉnh
- **Dataset**: CIFAR-100-LT (IR=100)
- **Architecture**: ResNet-32 với 3 experts (τ=0,1,2)
- **Training**: Standard training (200 epochs) với mixup calibration
- **Output**: 3 trained và calibrated expert models

### Stage 2: Optimize Plugin
- **Mục tiêu**: Tối ưu hóa plugin parameters cho rejection
- **Input**: Expert predictions từ Stage 1
- **Methods**: CS-plugin và Worst-group plugin
- **Output**: Optimized plugin parameters

### Stage 3: Evaluation và Comparison
- **Mục tiêu**: Đánh giá MoE-Plugin và so sánh với baselines
- **Protocol**: Risk-Coverage Curves và AURC metrics
- **Baselines**: BalPoE, Plugin_Single, Plugin_BalPoE_avg
- **Output**: Plots, tables, và comparison results

## Cấu trúc Files

```
moe_plugin_pipeline/
├── stage1_train_balpoe.py          # Stage 1: Train experts
├── stage2_optimize_plugin.py       # Stage 2: Optimize plugin
├── stage3_evaluate.py               # Stage 3: Evaluation
├── run_full_pipeline.py             # Run toàn bộ pipeline
├── run_full_pipeline.sh             # Shell script
├── configs/                         # Configuration files
│   ├── cifar100_ir100_balpoe.json  # Stage 1 config
│   ├── plugin_optimization.json     # Stage 2 config
│   └── evaluation_config.json       # Stage 3 config
├── plugin_methods/                  # Plugin optimization logic
│   └── plugin_optimizer.py
├── evaluation/                     # Evaluation framework
│   └── moe_plugin_evaluator.py
├── test_stage1.py                  # Test Stage 1
├── test_stage2.py                  # Test Stage 2
├── test_stage3.py                  # Test Stage 3
├── run_stage1.sh                   # Run Stage 1
├── run_stage2.sh                   # Run Stage 2
├── run_stage3.sh                   # Run Stage 3
├── STAGE1_GUIDE.md                 # Stage 1 guide
├── STAGE2_GUIDE.md                 # Stage 2 guide
├── STAGE3_GUIDE.md                 # Stage 3 guide
├── STAGE1_SUMMARY.md               # Stage 1 summary
├── STAGE2_SUMMARY.md               # Stage 2 summary
├── STAGE3_SUMMARY.md               # Stage 3 summary
├── requirements.txt                 # Dependencies
└── setup_environment.py           # Environment setup
```

## Cài đặt

### 1. Setup Environment
```bash
cd moe_plugin_pipeline
python setup_environment.py
```

### 2. Test Setup
```bash
# Test tất cả stages
python test_stage1.py
python test_stage2.py
python test_stage3.py
```

## Cách sử dụng

### 1. Chạy toàn bộ pipeline
```bash
# Chạy tất cả stages
python run_full_pipeline.py

# Hoặc sử dụng shell script
./run_full_pipeline.sh

# Chạy specific stages
python run_full_pipeline.py --stages 1 2 3
```

### 2. Chạy từng stage riêng lẻ

#### Stage 1: Train Experts
```bash
# Sử dụng script
./run_stage1.sh

# Hoặc chạy trực tiếp
python stage1_train_balpoe.py --config configs/cifar100_ir100_balpoe.json
```

#### Stage 2: Optimize Plugin
```bash
# Sử dụng script
./run_stage2.sh <experts_dir> <config_file>

# Hoặc chạy trực tiếp
python stage2_optimize_plugin.py \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --config configs/plugin_optimization.json
```

#### Stage 3: Evaluation
```bash
# Sử dụng script
./run_stage3.sh <plugin_checkpoint> <experts_dir> <config_file>

# Hoặc chạy trực tiếp
python stage3_evaluate.py \
    --plugin_checkpoint results/plugin_optimization/plugin_params.json \
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \
    --config configs/evaluation_config.json \
    --save_dir results/evaluation
```

## Configuration

### 1. Stage 1 Config (`configs/cifar100_ir100_balpoe.json`)
```json
{
    "name": "Imbalance_CIFAR100LT_IR100_BalPoE_Experts",
    "arch": {
        "type": "ResNet32Model",
        "args": {
            "num_classes": 100,
            "num_experts": 3
        }
    },
    "data_loader": {
        "type": "ImbalanceCIFAR100DataLoader",
        "args": {
            "data_dir": "./data/CIFAR-100",
            "batch_size": 128,
            "imb_factor": 0.01
        }
    },
    "optimizer": {
        "type": "SGD",
        "args": {
            "lr": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9
        }
    },
    "loss": {
        "type": "BSExpertLoss",
        "args": {
            "tau_list": [0, 1.0, 2.0]
        }
    },
    "trainer": {
        "epochs": 200,
        "save_dir": "checkpoints/balpoe_experts/cifar100_ir100"
    },
    "combiner": {
        "type": "Combiner",
        "mode": "mixup",
        "mixup": {
            "alpha": 0.4
        }
    }
}
```

### 2. Stage 2 Config (`configs/plugin_optimization.json`)
```json
{
    "name": "Plugin_Optimization",
    "dataset": {
        "name": "CIFAR-100-LT",
        "data_dir": "./data/CIFAR-100",
        "imbalance_ratio": 100
    },
    "group_definition": {
        "tail_threshold": 20,
        "head_threshold": 100
    },
    "cs_plugin": {
        "lambda_0_candidates": [1, 6, 11],
        "alpha_iterations": 10,
        "weight_search_space": {
            "min": 0.0,
            "max": 1.0,
            "step": 0.2
        }
    },
    "worst_group_plugin": {
        "iterations": 25,
        "step_size": 1.0,
        "initial_weights": [0.5, 0.5]
    },
    "output": {
        "save_dir": "results/plugin_optimization"
    }
}
```

### 3. Stage 3 Config (`configs/evaluation_config.json`)
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
        "tables": true
    }
}
```

## Output Files

### Stage 1 Output
- **Expert checkpoints**: `checkpoints/balpoe_experts/cifar100_ir100/`
- **Training logs**: Training progress và metrics
- **Config backup**: `config.json` trong checkpoint directory

### Stage 2 Output
- **Plugin parameters**: `results/plugin_optimization/plugin_params.json`
- **Optimization logs**: CS-plugin và Worst-group plugin results
- **Validation results**: Balanced error và worst-group error

### Stage 3 Output
- **Risk-Coverage Curves**: `results/evaluation/risk_coverage_curves.png`
- **AURC Comparison**: `results/evaluation/aurc_comparison.json`
- **Accuracy Comparison**: `results/evaluation/accuracy_comparison.json`
- **Detailed Results**: Evaluation metrics và analysis

## Key Features

### 1. Tuân thủ Papers
- **BalPoE**: Sử dụng đúng config và training protocol
- **Learning to Reject**: Risk-Coverage Curves, AURC, Group definition
- **MoE-Plugin**: Kết hợp hai approaches một cách hợp lý

### 2. Sử dụng Code từ BalPoE Gốc
- **Không tạo hàm mới**: Chỉ sử dụng functions có sẵn
- **Equivalent execution**: Chạy tương đương với repo gốc
- **Professional quality**: Code quality cao, dễ maintain

### 3. Comprehensive Evaluation
- **4 Baseline Methods**: Đầy đủ comparison
- **21 Cost Points**: Dense evaluation
- **2 Metrics**: Balanced Error và Worst-Group Error
- **Professional Output**: Plots, tables, JSON results

## Troubleshooting

### 1. Common Issues
- **Missing dependencies**: Chạy `python setup_environment.py`
- **Config errors**: Kiểm tra JSON format
- **Memory issues**: Giảm batch size
- **Path issues**: Kiểm tra relative paths

### 2. Debug Tips
- Chạy test scripts để kiểm tra setup
- Kiểm tra log output để debug
- Verify config files
- Check file permissions

### 3. Stage-specific Issues
- **Stage 1**: Kiểm tra data directory và config
- **Stage 2**: Kiểm tra expert checkpoints từ Stage 1
- **Stage 3**: Kiểm tra plugin checkpoint từ Stage 2

## Expected Results

### 1. AURC Ranking
1. **MoE-Plugin**: Lowest AURC (best performance)
2. **Plugin_BalPoE_avg**: Second best
3. **Plugin_Single**: Third
4. **BalPoE**: Highest AURC (no rejection capability)

### 2. Risk-Coverage Curves
- **MoE-Plugin**: Curve nằm dưới tất cả baselines
- **Plugin_BalPoE_avg**: Curve nằm dưới Plugin_Single
- **Plugin_Single**: Curve nằm dưới BalPoE
- **BalPoE**: Horizontal line (no rejection)

### 3. Performance Analysis
- **MoE-Plugin**: Best trade-off giữa accuracy và rejection
- **Plugin methods**: Better performance trên Few-shot classes
- **BalPoE**: Good baseline cho Head classes

## Kết luận

MoE-Plugin Pipeline cung cấp một framework toàn diện để:

1. **Train diverse experts**: Sử dụng BalPoE approach
2. **Optimize rejection**: Sử dụng Learning to Reject methods
3. **Evaluate comprehensively**: Professional evaluation protocol
4. **Compare fairly**: Multiple baselines và metrics

Pipeline sẵn sàng để chạy và đánh giá MoE-Plugin approach một cách chuyên nghiệp!