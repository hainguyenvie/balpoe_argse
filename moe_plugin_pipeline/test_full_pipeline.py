#!/usr/bin/env python3
"""
Test Full MoE-Plugin Pipeline
Kiểm tra setup và chạy thử toàn bộ pipeline
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_pipeline_structure():
    """Test cấu trúc pipeline"""
    print("🧪 Testing Pipeline Structure...")
    
    # Check required directories
    required_dirs = [
        "configs",
        "plugin_methods",
        "evaluation"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(__file__).parent / dir_name
        if not dir_path.exists():
            print(f"❌ Missing directory: {dir_name}")
            return False
        print(f"✅ Found directory: {dir_name}")
    
    # Check required files
    required_files = [
        "stage1_train_balpoe.py",
        "stage2_optimize_plugin.py",
        "stage3_evaluate.py",
        "run_full_pipeline.py",
        "run_full_pipeline.sh",
        "configs/cifar100_ir100_balpoe.json",
        "configs/plugin_optimization.json",
        "configs/evaluation_config.json",
        "plugin_methods/plugin_optimizer.py",
        "evaluation/moe_plugin_evaluator.py",
        "test_stage1.py",
        "test_stage2.py",
        "test_stage3.py",
        "requirements.txt",
        "setup_environment.py",
        "README.md"
    ]
    
    for file_name in required_files:
        file_path = Path(__file__).parent / file_name
        if not file_path.exists():
            print(f"❌ Missing file: {file_name}")
            return False
        print(f"✅ Found file: {file_name}")
    
    return True

def test_config_files():
    """Test config files"""
    print("\n🧪 Testing Config Files...")
    
    config_files = [
        "configs/cifar100_ir100_balpoe.json",
        "configs/plugin_optimization.json",
        "configs/evaluation_config.json"
    ]
    
    for config_file in config_files:
        config_path = Path(__file__).parent / config_file
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"✅ {config_file}: Valid JSON")
            
            # Check required fields
            if config_file == "configs/cifar100_ir100_balpoe.json":
                required_fields = ['name', 'arch', 'data_loader', 'optimizer', 'loss', 'trainer', 'combiner']
            elif config_file == "configs/plugin_optimization.json":
                required_fields = ['name', 'dataset', 'group_definition', 'cs_plugin', 'worst_group_plugin', 'output']
            elif config_file == "configs/evaluation_config.json":
                required_fields = ['name', 'dataset', 'group_definition', 'evaluation', 'output']
            
            for field in required_fields:
                if field not in config:
                    print(f"❌ {config_file}: Missing field '{field}'")
                    return False
                print(f"  ✅ Found: {field}")
            
        except json.JSONDecodeError as e:
            print(f"❌ {config_file}: Invalid JSON - {e}")
            return False
        except Exception as e:
            print(f"❌ {config_file}: Error - {e}")
            return False
    
    return True

def test_stage_imports():
    """Test imports của tất cả stages"""
    print("\n🧪 Testing Stage Imports...")
    
    stages = [
        ("Stage 1", "stage1_train_balpoe.py"),
        ("Stage 2", "stage2_optimize_plugin.py"),
        ("Stage 3", "stage3_evaluate.py")
    ]
    
    for stage_name, script_name in stages:
        try:
            script_path = Path(__file__).parent / script_name
            
            # Test basic imports
            import torch
            import numpy as np
            import json
            from pathlib import Path
            
            print(f"✅ {stage_name}: Core imports successful")
            
            # Test BalPoE imports
            sys.path.append(str(Path(__file__).parent.parent))
            import data_loader.data_loaders as module_data
            import model.model as module_arch
            from utils import seed_everything
            from parse_config import ConfigParser
            
            print(f"✅ {stage_name}: BalPoE imports successful")
            
        except ImportError as e:
            print(f"❌ {stage_name}: Import error - {e}")
            return False
        except Exception as e:
            print(f"❌ {stage_name}: Error - {e}")
            return False
    
    return True

def test_plugin_methods():
    """Test plugin methods"""
    print("\n🧪 Testing Plugin Methods...")
    
    try:
        from plugin_methods.plugin_optimizer import PluginParameters, PluginOptimizer
        
        # Test PluginParameters
        params = PluginParameters(
            lambda_0=6.0,
            alpha=0.5,
            expert_weights=[0.4, 0.3, 0.3],
            group_weights=[0.5, 0.5],
            rejection_threshold=0.5,
            balanced_error=0.3,
            worst_group_error=0.4
        )
        
        print(f"✅ PluginParameters: {params}")
        
        # Test PluginOptimizer
        optimizer = PluginOptimizer(params)
        print(f"✅ PluginOptimizer: Initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Plugin methods error: {e}")
        return False

def test_evaluation_framework():
    """Test evaluation framework"""
    print("\n🧪 Testing Evaluation Framework...")
    
    try:
        from evaluation.moe_plugin_evaluator import MoEPluginEvaluator
        
        # Test evaluator initialization
        evaluator = MoEPluginEvaluator(
            plugin_checkpoint="dummy_checkpoint.json",
            experts_dir="dummy_experts",
            config_path=str(Path(__file__).parent / "configs/evaluation_config.json"),
            seed=1
        )
        
        print(f"✅ MoEPluginEvaluator: Initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation framework error: {e}")
        return False

def test_full_pipeline_script():
    """Test full pipeline script"""
    print("\n🧪 Testing Full Pipeline Script...")
    
    try:
        from run_full_pipeline import run_stage1, run_stage2, run_stage3
        
        print("✅ Full pipeline functions: Available")
        
        # Test argument parsing
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--stages', nargs='+', default=['1', '2', '3'])
        parser.add_argument('--skip-tests', action='store_true')
        
        print("✅ Argument parsing: Available")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline script error: {e}")
        return False

def test_shell_scripts():
    """Test shell scripts"""
    print("\n🧪 Testing Shell Scripts...")
    
    shell_scripts = [
        "run_full_pipeline.sh",
        "run_stage1.sh",
        "run_stage2.sh",
        "run_stage3.sh"
    ]
    
    for script_name in shell_scripts:
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"❌ Missing shell script: {script_name}")
            return False
        
        # Check if script is executable
        if not script_path.stat().st_mode & 0o111:
            print(f"⚠️ Shell script not executable: {script_name}")
        else:
            print(f"✅ Shell script executable: {script_name}")
    
    return True

def test_documentation():
    """Test documentation files"""
    print("\n🧪 Testing Documentation...")
    
    doc_files = [
        "README.md",
        "STAGE1_GUIDE.md",
        "STAGE2_GUIDE.md",
        "STAGE3_GUIDE.md",
        "STAGE1_SUMMARY.md",
        "STAGE2_SUMMARY.md",
        "STAGE3_SUMMARY.md"
    ]
    
    for doc_file in doc_files:
        doc_path = Path(__file__).parent / doc_file
        
        if not doc_path.exists():
            print(f"❌ Missing documentation: {doc_file}")
            return False
        
        # Check file size (should not be empty)
        if doc_path.stat().st_size < 100:
            print(f"⚠️ Documentation file too small: {doc_file}")
        else:
            print(f"✅ Documentation file: {doc_file}")
    
    return True

def test_requirements():
    """Test requirements.txt"""
    print("\n🧪 Testing Requirements...")
    
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if not requirements_path.exists():
        print("❌ Missing requirements.txt")
        return False
    
    try:
        with open(requirements_path, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        print(f"✅ Requirements file: {len(requirements)} packages")
        
        # Check for essential packages
        essential_packages = ['torch', 'numpy', 'matplotlib', 'scikit-learn']
        for package in essential_packages:
            if any(package in req for req in requirements):
                print(f"  ✅ Found: {package}")
            else:
                print(f"  ⚠️ Missing: {package}")
        
        return True
        
    except Exception as e:
        print(f"❌ Requirements error: {e}")
        return False

def main():
    """Chạy tất cả tests cho full pipeline"""
    print("🚀 Testing Full MoE-Plugin Pipeline")
    print("=" * 60)
    
    tests = [
        test_pipeline_structure,
        test_config_files,
        test_stage_imports,
        test_plugin_methods,
        test_evaluation_framework,
        test_full_pipeline_script,
        test_shell_scripts,
        test_documentation,
        test_requirements
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
        print("✅ Full MoE-Plugin Pipeline sẵn sàng để chạy!")
        print("\n📋 Next Steps:")
        print("1. Chạy setup: python setup_environment.py")
        print("2. Test individual stages: python test_stage1.py, test_stage2.py, test_stage3.py")
        print("3. Run full pipeline: python run_full_pipeline.py")
        print("4. Analyze results: Check results/ directory")
    else:
        print("⚠️ Một số tests đã FAILED!")
        print("❌ Cần kiểm tra lại pipeline!")

if __name__ == '__main__':
    main()
