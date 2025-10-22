#!/usr/bin/env python3
"""
Test script để kiểm tra Stage 2 có hoạt động đúng không
Kiểm tra config, imports, và các dependencies
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Kiểm tra các import cần thiết"""
    print("🔍 Testing imports...")
    
    try:
        # Test BalPoE imports
        import data_loader.data_loaders as module_data
        from utils import seed_everything
        from parse_config import ConfigParser
        import model.model as module_arch
        
        print("✅ All BalPoE imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from BalPoE root directory")
        return False

def test_config():
    """Kiểm tra config file"""
    print("🔍 Testing config file...")
    
    config_path = Path("moe_plugin_pipeline/configs/plugin_optimization.json")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Kiểm tra các key quan trọng
        required_keys = [
            'dataset', 'group_definition', 'cs_plugin', 
            'worst_group_plugin', 'experts', 'output'
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing key: {key}")
                return False
        
        # Kiểm tra CS-plugin configuration
        if config['cs_plugin']['lambda_0_candidates'] != [1, 6, 11]:
            print(f"❌ Wrong lambda_0_candidates: {config['cs_plugin']['lambda_0_candidates']}")
            return False
        
        if config['cs_plugin']['alpha_search_iterations'] != 10:
            print(f"❌ Wrong alpha_search_iterations: {config['cs_plugin']['alpha_search_iterations']}")
            return False
        
        # Kiểm tra Worst-group plugin configuration
        if config['worst_group_plugin']['max_iterations'] != 25:
            print(f"❌ Wrong max_iterations: {config['worst_group_plugin']['max_iterations']}")
            return False
        
        if config['worst_group_plugin']['step_size'] != 1.0:
            print(f"❌ Wrong step_size: {config['worst_group_plugin']['step_size']}")
            return False
        
        # Kiểm tra group definition
        if config['group_definition']['tail_threshold'] != 20:
            print(f"❌ Wrong tail_threshold: {config['group_definition']['tail_threshold']}")
            return False
        
        print("✅ Config file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_experts_directory():
    """Kiểm tra thư mục experts"""
    print("🔍 Testing experts directory...")
    
    experts_dir = Path("checkpoints/balpoe_experts/cifar100_ir100")
    if not experts_dir.exists():
        print(f"⚠️  Experts directory not found: {experts_dir}")
        print("💡 Please run Stage 1 first to create expert checkpoints")
        return False
    else:
        # Check for checkpoint files
        checkpoint_files = list(experts_dir.glob("*.pth"))
        if not checkpoint_files:
            print(f"❌ No checkpoint files found in {experts_dir}")
            return False
        
        print(f"✅ Found {len(checkpoint_files)} checkpoint files")
        return True

def test_data_directory():
    """Kiểm tra thư mục data"""
    print("🔍 Testing data directory...")
    
    data_dir = Path("data/CIFAR-100")
    if not data_dir.exists():
        print(f"⚠️  Data directory not found: {data_dir}")
        print("💡 This is OK - dataset will be created automatically")
        return True
    else:
        print("✅ Data directory exists")
        return True

def test_model_creation():
    """Kiểm tra tạo model"""
    print("🔍 Testing model creation...")
    
    try:
        import model.model as module_arch
        
        # Test model creation
        model = getattr(module_arch, "ResNet32Model")(
            num_classes=100,
            reduce_dimension=False,
            use_norm=True,
            returns_feat=True,
            num_experts=3
        )
        
        print(f"✅ Model created: {type(model).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_data_loader_creation():
    """Kiểm tra tạo data loader"""
    print("🔍 Testing data loader creation...")
    
    try:
        import data_loader.data_loaders as module_data
        
        # Test data loader creation
        data_loader = getattr(module_data, "ImbalanceCIFAR100DataLoader")(
            data_dir="./data/CIFAR-100",
            batch_size=256,
            shuffle=False,
            num_workers=4,
            imb_factor=0.01,
            randaugm=False,
            training=False
        )
        
        print(f"✅ Data loader created: {type(data_loader).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Data loader creation error: {e}")
        return False

def main():
    """Chạy tất cả tests"""
    print("🧪 TESTING STAGE 2 SETUP")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Experts Directory", test_experts_directory),
        ("Data Directory", test_data_directory),
        ("Model Creation", test_model_creation),
        ("Data Loader Creation", test_data_loader_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Stage 2 is ready to run")
        print("\n🚀 To run Stage 2:")
        print("   python moe_plugin_pipeline/stage2_optimize_plugin.py \\")
        print("       --config moe_plugin_pipeline/configs/plugin_optimization.json \\")
        print("       --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \\")
        print("       --seed 1")
    else:
        print("❌ SOME TESTS FAILED!")
        print("💡 Please fix the issues above before running Stage 2")
    
    return all_passed

if __name__ == "__main__":
    main()
