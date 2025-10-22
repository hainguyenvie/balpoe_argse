#!/usr/bin/env python3
"""
Setup Environment for MoE-Plugin Pipeline
Thiết lập môi trường code và dependencies
"""

import subprocess
import sys
from pathlib import Path
import os


def run_command(cmd, description):
    """Chạy command và in kết quả"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def setup_environment():
    """Thiết lập môi trường cho MoE-Plugin pipeline"""
    
    print("🚀 SETTING UP MOE-PLUGIN PIPELINE ENVIRONMENT")
    print("="*60)
    
    # 1. Create directory structure
    print("\n📁 Creating directory structure...")
    directories = [
        "data",
        "checkpoints/balpoe_experts",
        "checkpoints/plugin_optimized", 
        "results/evaluation",
        "logs",
        "notebooks"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {dir_path}")
    
    # 2. Install requirements
    print("\n📦 Installing requirements...")
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        if not run_command("pip install -r requirements.txt", "Installing Python packages"):
            print("❌ Failed to install requirements. Please install manually:")
            print("   pip install -r requirements.txt")
            return False
    else:
        print("❌ requirements.txt not found in root directory")
        print("📝 Please ensure requirements.txt exists in the root directory")
        return False
    
    # 3. Create symlinks to original BalPoE code
    print("\n🔗 Creating symlinks to BalPoE code...")
    
    # Check if we're in the BalPoE directory
    if Path("train.py").exists() and Path("model").exists():
        print("  ✅ Found BalPoE code in current directory")
        
        # Create symlinks
        symlinks = [
            ("model", "moe_plugin_pipeline/model"),
            ("data_loader", "moe_plugin_pipeline/data_loader"),
            ("trainer", "moe_plugin_pipeline/trainer"),
            ("utils", "moe_plugin_pipeline/utils"),
            ("base", "moe_plugin_pipeline/base"),
            ("parse_config.py", "moe_plugin_pipeline/parse_config.py")
        ]
        
        for src, dst in symlinks:
            if Path(src).exists():
                if Path(dst).exists() or Path(dst).is_symlink():
                    Path(dst).unlink()
                Path(dst).symlink_to(Path(src).absolute())
                print(f"  ✅ Linked: {src} -> {dst}")
            else:
                print(f"  ⚠️  Not found: {src}")
    else:
        print("  ⚠️  BalPoE code not found in current directory")
        print("  📝 Please copy BalPoE files manually or run from BalPoE directory")
    
    # 4. Create data directory structure
    print("\n📊 Setting up data directories...")
    data_dirs = [
        "data/CIFAR-100",
        "data/ImageNet-LT", 
        "data/iNaturalist-2018"
    ]
    
    for data_dir in data_dirs:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {data_dir}")
    
    # 5. Create example scripts
    print("\n📝 Creating example scripts...")
    
    # Example training script
    example_script = """#!/bin/bash
# Example: Run MoE-Plugin Pipeline

# Stage 1: Train BalPoE experts
python moe_plugin_pipeline/stage1_train_balpoe.py \\
    --config moe_plugin_pipeline/configs/cifar100_ir100_balpoe.json \\
    --seed 1

# Stage 2: Optimize plugin parameters  
python moe_plugin_pipeline/stage2_optimize_plugin.py \\
    --config moe_plugin_pipeline/configs/plugin_optimization.json \\
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \\
    --seed 1

# Stage 3: Evaluate pipeline
python moe_plugin_pipeline/stage3_evaluate.py \\
    --plugin_checkpoint checkpoints/plugin_optimized/optimized_parameters.json \\
    --experts_dir checkpoints/balpoe_experts/cifar100_ir100 \\
    --config moe_plugin_pipeline/configs/plugin_optimization.json \\
    --save_dir results/evaluation \\
    --seed 1
"""
    
    with open("run_example.sh", "w") as f:
        f.write(example_script)
    os.chmod("run_example.sh", 0o755)
    print("  ✅ Created: run_example.sh")
    
    # 6. Create Jupyter notebook for analysis
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MoE-Plugin Pipeline Analysis\\n",
    "\\n",
    "This notebook provides analysis tools for the MoE-Plugin pipeline results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\\n",
    "import matplotlib.pyplot as plt\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "\\n",
    "# Load evaluation results\\n",
    "with open('results/evaluation/evaluation_results.json', 'r') as f:\\n",
    "    results = json.load(f)\\n",
    "\\n",
    "# Create comparison dataframe\\n",
    "df = pd.DataFrame(results)\\n",
    "print(df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    with open("notebooks/analysis.ipynb", "w") as f:
        f.write(notebook_content)
    print("  ✅ Created: notebooks/analysis.ipynb")
    
    print("\n" + "="*60)
    print("🎉 ENVIRONMENT SETUP COMPLETED!")
    print("="*60)
    print("📁 Directory structure created")
    print("📦 Dependencies installed")
    print("🔗 Symlinks created (if BalPoE code available)")
    print("📝 Example scripts created")
    print("\n🚀 Next steps:")
    print("1. Download datasets to data/ directory")
    print("2. Run: python moe_plugin_pipeline/run_full_pipeline.py")
    print("3. Or run individual stages as needed")
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    setup_environment()
