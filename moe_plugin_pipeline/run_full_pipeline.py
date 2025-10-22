#!/usr/bin/env python3
"""
Run Full MoE-Plugin Pipeline
Chạy toàn bộ pipeline từ Stage 1 đến Stage 3
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_stage1():
    """Chạy Stage 1: Train BalPoE Experts"""
    print("🚀 Stage 1: Training BalPoE Experts")
    print("=" * 50)
    
    # Check if Stage 1 config exists
    config_path = Path(__file__).parent / "configs/cifar100_ir100_balpoe.json"
    if not config_path.exists():
        print(f"❌ Stage 1 config not found: {config_path}")
        return False
    
    # Run Stage 1
    cmd = [
        "python", "stage1_train_balpoe.py",
        "--config", str(config_path)
    ]
    
    print(f"🔍 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("✅ Stage 1 completed successfully!")
        return True
    else:
        print("❌ Stage 1 failed!")
        return False

def run_stage2():
    """Chạy Stage 2: Optimize Plugin"""
    print("\n🚀 Stage 2: Optimizing Plugin")
    print("=" * 50)
    
    # Check if Stage 2 config exists
    config_path = Path(__file__).parent / "configs/plugin_optimization.json"
    if not config_path.exists():
        print(f"❌ Stage 2 config not found: {config_path}")
        return False
    
    # Check if experts exist
    experts_dir = Path(__file__).parent.parent / "checkpoints/balpoe_experts/cifar100_ir100"
    if not experts_dir.exists():
        print(f"❌ Experts directory not found: {experts_dir}")
        return False
    
    # Run Stage 2
    cmd = [
        "python", "stage2_optimize_plugin.py",
        "--experts_dir", str(experts_dir),
        "--config", str(config_path)
    ]
    
    print(f"🔍 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("✅ Stage 2 completed successfully!")
        return True
    else:
        print("❌ Stage 2 failed!")
        return False

def run_stage3():
    """Chạy Stage 3: Evaluation và Comparison"""
    print("\n🚀 Stage 3: Evaluation và Comparison")
    print("=" * 50)
    
    # Check if Stage 3 config exists
    config_path = Path(__file__).parent / "configs/evaluation_config.json"
    if not config_path.exists():
        print(f"❌ Stage 3 config not found: {config_path}")
        return False
    
    # Check if plugin checkpoint exists
    plugin_checkpoint = Path(__file__).parent / "results/plugin_optimization/plugin_params.json"
    if not plugin_checkpoint.exists():
        print(f"❌ Plugin checkpoint not found: {plugin_checkpoint}")
        return False
    
    # Check if experts exist
    experts_dir = Path(__file__).parent.parent / "checkpoints/balpoe_experts/cifar100_ir100"
    if not experts_dir.exists():
        print(f"❌ Experts directory not found: {experts_dir}")
        return False
    
    # Run Stage 3
    cmd = [
        "python", "stage3_evaluate.py",
        "--plugin_checkpoint", str(plugin_checkpoint),
        "--experts_dir", str(experts_dir),
        "--config", str(config_path),
        "--save_dir", "results/evaluation"
    ]
    
    print(f"🔍 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("✅ Stage 3 completed successfully!")
        return True
    else:
        print("❌ Stage 3 failed!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Full MoE-Plugin Pipeline')
    parser.add_argument('--stages', nargs='+', default=['1', '2', '3'],
                       help='Stages to run (default: all stages)')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip test scripts')
    
    args = parser.parse_args()
    
    print("🚀 MoE-Plugin Pipeline")
    print("=" * 60)
    print(f"📋 Stages to run: {args.stages}")
    print(f"🧪 Skip tests: {args.skip_tests}")
    print("=" * 60)
    
    # Run test scripts if not skipped
    if not args.skip_tests:
        print("\n🧪 Running test scripts...")
        
        test_scripts = [
            ("Stage 1", "test_stage1.py"),
            ("Stage 2", "test_stage2.py"),
            ("Stage 3", "test_stage3.py")
        ]
        
        for stage_name, test_script in test_scripts:
            if stage_name.split()[-1] in args.stages:
                print(f"\n🔍 Testing {stage_name}...")
                cmd = ["python", test_script]
                result = subprocess.run(cmd, cwd=Path(__file__).parent)
                
                if result.returncode == 0:
                    print(f"✅ {stage_name} tests passed!")
                else:
                    print(f"❌ {stage_name} tests failed!")
                    return False
    
    # Run stages
    stage_functions = {
        '1': run_stage1,
        '2': run_stage2,
        '3': run_stage3
    }
    
    for stage in args.stages:
        if stage in stage_functions:
            success = stage_functions[stage]()
            if not success:
                print(f"\n❌ Pipeline failed at Stage {stage}!")
                return False
        else:
            print(f"❌ Unknown stage: {stage}")
            return False
    
    print("\n" + "=" * 60)
    print("🎉 MoE-Plugin Pipeline completed successfully!")
    print("📁 Results saved to:")
    print("  - Stage 1: checkpoints/balpoe_experts/cifar100_ir100/")
    print("  - Stage 2: results/plugin_optimization/")
    print("  - Stage 3: results/evaluation/")
    print("\n📊 Ready for analysis and comparison!")

if __name__ == '__main__':
    main()