#!/usr/bin/env python3
"""
Run Improved Stage 2: Analysis + Optimization
This script runs expert quality analysis and improved optimization
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def main():
    print("🚀 Running Improved Stage 2: Analysis + Optimization")
    print("=" * 60)
    
    # Check if config exists
    config_file = "moe_plugin_pipeline/configs/plugin_optimization.json"
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Step 1: Analyze expert quality
    print("📊 Step 1: Analyzing Expert Quality...")
    print("-" * 40)
    
    expert_analysis_cmd = f"""
    python moe_plugin_pipeline/debug_expert_quality.py \
        --experts_dir "checkpoints/balpoe_experts/cifar100_ir100/models/Imbalance_CIFAR100LT_IR100_BalPoE_Experts/1022_153647" \
        --config "{config_file}"
    """
    
    if not run_command(expert_analysis_cmd, "Expert quality analysis"):
        print("❌ Expert quality analysis failed")
        return False
    
    print("\n📊 Step 2: Running Improved Optimization...")
    print("-" * 40)
    
    improved_optimization_cmd = f"""
    python moe_plugin_pipeline/improved_optimization.py \
        --config "{config_file}"
    """
    
    if not run_command(improved_optimization_cmd, "Improved optimization"):
        print("❌ Improved optimization failed")
        return False
    
    print("\n✅ Improved Stage 2 completed!")
    print("📁 Results saved to: checkpoints/plugin_optimized/")
    print("🔧 Next step: python moe_plugin_pipeline/stage3_evaluate.py --plugin_checkpoint checkpoints/plugin_optimized/improved_optimized_parameters.json")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
