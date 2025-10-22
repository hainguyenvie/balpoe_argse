#!/usr/bin/env python3
"""
Test Requirements File
Kiểm tra file requirements.txt tổng hợp
"""

import sys
import subprocess
from pathlib import Path

def test_requirements_file():
    """Test file requirements.txt"""
    print("🧪 Testing Requirements File...")
    
    requirements_path = Path("requirements.txt")
    
    if not requirements_path.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print(f"✅ Found: {requirements_path}")
    
    # Check file content
    with open(requirements_path, 'r') as f:
        content = f.read()
    
    # Check for essential packages
    essential_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pandas',
        'tqdm',
        'tensorboard',
        'Pillow',
        'pyyaml',
        'h5py',
        'lightning',
        'easydict'
    ]
    
    missing_packages = []
    for package in essential_packages:
        if package not in content:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        return False
    
    print(f"✅ All essential packages found: {len(essential_packages)}")
    
    # Check for Python 3.8 compatibility
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor == 8:
        print("✅ Compatible with Python 3.8")
    else:
        print(f"⚠️  Current Python version: {python_version.major}.{python_version.minor}")
        print("📝 Requirements optimized for Python 3.8")
    
    return True

def test_package_installation():
    """Test package installation"""
    print("\n🧪 Testing Package Installation...")
    
    try:
        # Test core packages
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import sklearn
        import tqdm
        import tensorboard
        from PIL import Image
        import yaml
        import h5py
        import lightning
        import easydict
        
        print("✅ Core packages imported successfully")
        
        # Test additional packages
        import scipy
        import seaborn as sns
        import plotly
        
        print("✅ Additional packages imported successfully")
        
        # Test development packages
        import jupyter
        import ipykernel
        import black
        import flake8
        
        print("✅ Development packages imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📝 Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_version_compatibility():
    """Test version compatibility"""
    print("\n🧪 Testing Version Compatibility...")
    
    try:
        import torch
        import numpy as np
        import matplotlib
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        
        # Check PyTorch version compatibility
        torch_version = torch.__version__
        if torch_version.startswith('1.') or torch_version.startswith('2.'):
            print("✅ PyTorch version compatible")
        else:
            print(f"⚠️  PyTorch version: {torch_version}")
        
        # Check NumPy version compatibility
        numpy_version = np.__version__
        if numpy_version.startswith('1.'):
            print("✅ NumPy version compatible")
        else:
            print(f"⚠️  NumPy version: {numpy_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Version compatibility error: {e}")
        return False

def main():
    """Chạy tất cả tests cho requirements"""
    print("🚀 Testing Requirements File")
    print("=" * 50)
    
    tests = [
        test_requirements_file,
        test_package_installation,
        test_version_compatibility
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
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All requirements tests passed!")
        print("✅ Requirements file is ready to use!")
        print("\n📋 Next Steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Run MoE-Plugin pipeline: python moe_plugin_pipeline/run_full_pipeline.py")
        print("3. Check results in results/ directory")
    else:
        print("⚠️ Some requirements tests failed!")
        print("❌ Please check requirements.txt and install missing packages!")

if __name__ == '__main__':
    main()
