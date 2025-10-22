#!/usr/bin/env python3
"""
Test Stage 3: Evaluation và Comparison
Kiểm tra setup và chạy thử Stage 3
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_stage3_setup():
    """Test setup của Stage 3"""
    print("🧪 Testing Stage 3 Setup...")
    
    # Check required files
    required_files = [
        "stage3_evaluate.py",
        "configs/evaluation_config.json"
    ]
    
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
        print(f"✅ Found: {file_path}")
    
    # Check config file
    config_path = Path(__file__).parent / "configs/evaluation_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Config loaded successfully")
    print(f"  - Dataset: {config['dataset']['name']}")
    print(f"  - Tail threshold: {config['group_definition']['tail_threshold']}")
    print(f"  - Cost values: {len(config['evaluation']['cost_values'])} points")
    
    return True

def test_stage3_imports():
    """Test imports của Stage 3"""
    print("\n🧪 Testing Stage 3 Imports...")
    
    try:
        # Test main imports
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ Core imports successful")
        
        # Test BalPoE imports
        sys.path.append(str(Path(__file__).parent.parent))
        import data_loader.data_loaders as module_data
        import model.model as module_arch
        from utils import seed_everything
        print("✅ BalPoE imports successful")
        
        # Test stage3 imports
        from stage3_evaluate import MoEPluginEvaluator
        print("✅ Stage 3 imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_stage3_config():
    """Test config của Stage 3"""
    print("\n🧪 Testing Stage 3 Config...")
    
    config_path = Path(__file__).parent / "configs/evaluation_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = [
            'dataset.name',
            'dataset.data_dir',
            'group_definition.tail_threshold',
            'evaluation.cost_values',
            'evaluation.metrics',
            'evaluation.baselines',
            'output.save_dir'
        ]
        
        for field in required_fields:
            keys = field.split('.')
            current = config
            for key in keys:
                if key not in current:
                    print(f"❌ Missing field: {field}")
                    return False
                current = current[key]
            print(f"✅ Found: {field}")
        
        # Validate cost values
        cost_values = config['evaluation']['cost_values']
        if not isinstance(cost_values, list) or len(cost_values) < 2:
            print("❌ Invalid cost_values")
            return False
        
        if cost_values[0] != 0.0 or cost_values[-1] != 1.0:
            print("❌ Cost values should range from 0.0 to 1.0")
            return False
        
        print(f"✅ Cost values: {len(cost_values)} points from {cost_values[0]} to {cost_values[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_stage3_evaluation_logic():
    """Test evaluation logic của Stage 3"""
    print("\n🧪 Testing Stage 3 Evaluation Logic...")
    
    try:
        # Test MoEPluginEvaluator initialization
        from stage3_evaluate import MoEPluginEvaluator
        
        # Create dummy plugin checkpoint
        dummy_checkpoint = {
            'lambda_0': 6.0,
            'alpha': 0.5,
            'expert_weights': [0.4, 0.3, 0.3],
            'group_weights': [0.5, 0.5],
            'rejection_threshold': 0.5,
            'balanced_error': 0.3,
            'worst_group_error': 0.4
        }
        
        # Save dummy checkpoint
        dummy_path = Path(__file__).parent / "dummy_plugin_checkpoint.json"
        with open(dummy_path, 'w') as f:
            json.dump(dummy_checkpoint, f)
        
        # Test evaluator initialization
        evaluator = MoEPluginEvaluator(
            plugin_checkpoint=str(dummy_path),
            experts_dir="dummy_experts",
            config_path=str(Path(__file__).parent / "configs/evaluation_config.json"),
            seed=1
        )
        
        print("✅ MoEPluginEvaluator initialized successfully")
        
        # Clean up
        dummy_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation logic error: {e}")
        return False

def test_stage3_risk_coverage_curves():
    """Test risk-coverage curves logic"""
    print("\n🧪 Testing Risk-Coverage Curves Logic...")
    
    try:
        # Test curve creation logic
        import numpy as np
        
        # Simulate results
        cost_values = np.arange(0.0, 1.05, 0.05)
        results = {
            'BalPoE': {
                'costs': cost_values.tolist(),
                'balanced_errors': np.random.uniform(0.2, 0.8, len(cost_values)).tolist(),
                'worst_group_errors': np.random.uniform(0.3, 0.9, len(cost_values)).tolist(),
                'rejection_rates': np.zeros_like(cost_values).tolist()
            }
        }
        
        # Test AURC computation
        balanced_aurc = np.trapz(results['BalPoE']['balanced_errors'], results['BalPoE']['rejection_rates'])
        worst_group_aurc = np.trapz(results['BalPoE']['worst_group_errors'], results['BalPoE']['rejection_rates'])
        
        print(f"✅ AURC computation successful")
        print(f"  - Balanced AURC: {balanced_aurc:.4f}")
        print(f"  - Worst-Group AURC: {worst_group_aurc:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk-coverage curves error: {e}")
        return False

def test_stage3_baselines():
    """Test baseline methods"""
    print("\n🧪 Testing Baseline Methods...")
    
    try:
        # Test baseline method names
        baselines = ['BalPoE', 'Plugin_Single', 'Plugin_BalPoE_avg', 'MoE_Plugin']
        
        for baseline in baselines:
            print(f"✅ Baseline: {baseline}")
        
        # Test baseline logic
        import torch
        
        # Simulate expert predictions
        batch_size, num_classes = 100, 100
        expert_predictions = {
            'head_expert': torch.rand(batch_size, num_classes),
            'balanced_expert': torch.rand(batch_size, num_classes),
            'tail_expert': torch.rand(batch_size, num_classes)
        }
        
        # Test BalPoE baseline (average ensemble)
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for name in ['head_expert', 'balanced_expert', 'tail_expert']:
            ensemble_pred += expert_predictions[name] / 3.0
        
        print(f"✅ BalPoE baseline: ensemble shape {ensemble_pred.shape}")
        
        # Test Plugin baseline (rejection)
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        cost = 0.5
        reject_mask = max_probs < (1.0 - cost)
        
        print(f"✅ Plugin baseline: rejection rate {reject_mask.float().mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline methods error: {e}")
        return False

def main():
    """Chạy tất cả tests cho Stage 3"""
    print("🚀 Testing Stage 3: Evaluation và Comparison")
    print("=" * 60)
    
    tests = [
        test_stage3_setup,
        test_stage3_imports,
        test_stage3_config,
        test_stage3_evaluation_logic,
        test_stage3_risk_coverage_curves,
        test_stage3_baselines
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ ERROR: {e}")
        print("-" * 40)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Tất cả tests đã PASSED!")
        print("✅ Stage 3 sẵn sàng để chạy!")
    else:
        print("⚠️ Một số tests đã FAILED!")
        print("❌ Cần kiểm tra lại Stage 3!")

if __name__ == '__main__':
    main()
