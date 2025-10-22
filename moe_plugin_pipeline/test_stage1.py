#!/usr/bin/env python3
"""
Test script để kiểm tra Stage 1 có hoạt động đúng không
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
        from utils import seed_everything, write_json, parse_tau_list, learning_rate_scheduler
        from parse_config import ConfigParser
        import data_loader.data_loaders as module_data
        import model.loss as module_loss
        import model.metric as module_metric
        import model.model as module_arch
        import utils.combiner as module_combiner
        from trainer import Trainer
        
        print("✅ All BalPoE imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from BalPoE root directory")
        return False

def test_config():
    """Kiểm tra config file"""
    print("🔍 Testing config file...")
    
    config_path = Path("moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Kiểm tra các key quan trọng
        required_keys = [
            'arch', 'data_loader', 'optimizer', 'loss', 
            'lr_scheduler', 'trainer', 'combiner'
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing key: {key}")
                return False
        
        # Kiểm tra expert configuration
        if config['arch']['args']['num_experts'] != 3:
            print(f"❌ Wrong num_experts: {config['arch']['args']['num_experts']}")
            return False
        
        if config['loss']['args']['tau_list'] != [0, 1.0, 2.0]:
            print(f"❌ Wrong tau_list: {config['loss']['args']['tau_list']}")
            return False
        
        if config['combiner']['mixup']['alpha'] != 0.4:
            print(f"❌ Wrong mixup alpha: {config['combiner']['mixup']['alpha']}")
            return False
        
        print("✅ Config file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

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
        from parse_config import ConfigParser
        
        config_path = "moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ConfigParser(config_dict)
        
        # Test model creation
        import model.model as module_arch
        model = config.init_obj('arch', module_arch)
        
        print(f"✅ Model created: {type(model).__name__}")
        print(f"✅ Number of experts: {config['arch']['args']['num_experts']}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_loss_creation():
    """Kiểm tra tạo loss function"""
    print("🔍 Testing loss function creation...")
    
    try:
        from parse_config import ConfigParser
        import model.loss as module_loss
        
        config_path = "moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ConfigParser(config_dict)
        
        # Test loss creation
        loss_class = getattr(module_loss, config["loss"]["type"])
        extra_parameters = {}
        
        if hasattr(loss_class, "require_num_experts") and loss_class.require_num_experts:
            extra_parameters["num_experts"] = config["arch"]["args"]["num_experts"]
        
        # Mock cls_num_list for testing
        cls_num_list = [5000] * 100  # Mock class distribution
        
        criterion = config.init_obj('loss', module_loss, 
                                 cls_num_list=cls_num_list, 
                                 **extra_parameters)
        
        print(f"✅ Loss function created: {type(criterion).__name__}")
        print(f"✅ Tau list: {config['loss']['args']['tau_list']}")
        return True
        
    except Exception as e:
        print(f"❌ Loss creation error: {e}")
        return False

def main():
    """Chạy tất cả tests"""
    print("🧪 TESTING STAGE 1 SETUP")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Data Directory", test_data_directory),
        ("Model Creation", test_model_creation),
        ("Loss Creation", test_loss_creation)
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
        print("✅ Stage 1 is ready to run")
        print("\n🚀 To run Stage 1:")
        print("   python moe_plugin_pipeline/stage1_train_balpoe.py \\")
        print("       -c moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json \\")
        print("       -s 1")
    else:
        print("❌ SOME TESTS FAILED!")
        print("💡 Please fix the issues above before running Stage 1")
    
    return all_passed

if __name__ == "__main__":
    main()
