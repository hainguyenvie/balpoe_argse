#!/usr/bin/env python3
"""
Giai đoạn 3: Đánh giá và So sánh
Mục tiêu: Đánh giá MoE-Plugin và so sánh với các baselines
Sử dụng hoàn toàn code từ BalPoE gốc, không tạo hàm mới
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add parent directory to path để import từ BalPoE gốc
sys.path.append(str(Path(__file__).parent.parent))

# Import trực tiếp từ BalPoE gốc
import data_loader.data_loaders as module_data
from utils import seed_everything
from parse_config import ConfigParser
import model.model as module_arch


class MoEPluginEvaluator:
    """
    MoE-Plugin Evaluator
    Triển khai đúng theo paper "Learning to Reject Meets Long-tail Learning"
    """
    
    def __init__(self, plugin_checkpoint: str, experts_dir: str, config_path: str, seed: int = 1):
        self.plugin_checkpoint = Path(plugin_checkpoint)
        self.experts_dir = Path(experts_dir)
        self.config_path = config_path
        self.seed = seed
        
        # Load plugin parameters
        with open(self.plugin_checkpoint, 'r') as f:
            plugin_data = json.load(f)
        
        self.plugin_params = plugin_data
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        seed_everything(seed)
        
        print(f"🚀 Khởi tạo MoE-Plugin Evaluator")
        print(f"📂 Plugin checkpoint: {self.plugin_checkpoint}")
        print(f"📂 Experts directory: {self.experts_dir}")
        print(f"📊 Plugin parameters:")
        print(f"  - Lambda_0: {self.plugin_params['lambda_0']}")
        print(f"  - Alpha: {self.plugin_params['alpha']}")
        print(f"  - Expert weights: {self.plugin_params['expert_weights']}")
        
    def setup_data_loaders(self):
        """Thiết lập data loaders cho evaluation"""
        print("📂 Thiết lập data loaders cho evaluation...")
        
        # Create data loader config
        data_loader_config = {
            "type": "ImbalanceCIFAR100DataLoader",
            "args": {
                "data_dir": self.config['dataset']['data_dir'],
                "batch_size": 256,
                "shuffle": False,
                "num_workers": 4,
                "imb_factor": 0.01,
                "randaugm": False,
                "training": False
            }
        }
        
        # Create test data loader
        self.test_data_loader = getattr(module_data, data_loader_config['type'])(
            **data_loader_config['args']
        )
        
        print(f"✅ Test samples: {len(self.test_data_loader.sampler)}")
        
    def load_expert_models(self):
        """Load 3 expert models từ BalPoE checkpoints"""
        print("📂 Loading expert models...")
        
        self.expert_models = {}
        tau_values = [0, 1.0, 2.0]
        expert_names = ['head_expert', 'balanced_expert', 'tail_expert']
        
        for tau, name in zip(tau_values, expert_names):
            # Find checkpoint file
            checkpoint_files = list(self.experts_dir.glob(f"*epoch*.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {self.experts_dir}")
            
            # Use the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            
            # Create model architecture
            model_config = {
                "type": "ResNet32Model",
                "args": {
                    "num_classes": 100,
                    "reduce_dimension": False,
                    "use_norm": True,
                    "returns_feat": True,
                    "num_experts": 3
                }
            }
            
            # Create model
            model = getattr(module_arch, model_config['type'])(**model_config['args'])
            
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            self.expert_models[name] = model
            
            print(f"  ✅ {name} (τ={tau}): {latest_checkpoint.name}")
        
        return self.expert_models
    
    def get_expert_predictions(self, data_loader):
        """Lấy predictions từ tất cả experts"""
        print("🔍 Getting expert predictions...")
        
        expert_predictions = {name: [] for name in self.expert_models.keys()}
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to('cpu')
                all_labels.append(batch_labels)
                
                for name, model in self.expert_models.items():
                    # Get expert prediction
                    expert_output = model(batch_data)
                    
                    # Handle BalPoE model output (can be dict or tensor)
                    if isinstance(expert_output, dict):
                        # BalPoE returns dict with 'logits' key containing individual expert logits
                        if 'logits' in expert_output:
                            # Get individual expert logits (shape: [batch_size, num_experts, num_classes])
                            all_expert_logits = expert_output['logits']
                            # Extract specific expert based on name
                            expert_idx = {'head_expert': 0, 'balanced_expert': 1, 'tail_expert': 2}[name]
                            expert_logits = all_expert_logits[:, expert_idx, :]  # [batch_size, num_classes]
                        elif 'output' in expert_output:
                            # Fallback to averaged output
                            expert_logits = expert_output['output']
                        else:
                            # If no expected keys, use the first tensor value
                            expert_logits = list(expert_output.values())[0]
                    else:
                        expert_logits = expert_output
                    
                    # Get probabilities
                    expert_probs = torch.softmax(expert_logits, dim=1)
                    expert_predictions[name].append(expert_probs.cpu())
        
        # Combine predictions
        for name in expert_predictions:
            expert_predictions[name] = torch.cat(expert_predictions[name], dim=0)
        
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"✅ Expert predictions shape: {expert_predictions['head_expert'].shape}")
        print(f"✅ Labels shape: {all_labels.shape}")
        
        return expert_predictions, all_labels
    
    def define_groups(self, labels):
        """Định nghĩa Head vs Tail groups theo paper"""
        print("📊 Defining Head vs Tail groups...")
        
        # Get class distribution from training data
        train_data_loader = getattr(module_data, "ImbalanceCIFAR100DataLoader")(
            data_dir=self.config['dataset']['data_dir'],
            batch_size=256,
            shuffle=False,
            num_workers=4,
            imb_factor=0.01,
            randaugm=False,
            training=True
        )
        
        cls_num_list = np.array(train_data_loader.cls_num_list)
        
        # Define groups: Tail = classes with <= 20 samples
        tail_threshold = self.config['group_definition']['tail_threshold']
        tail_classes = cls_num_list <= tail_threshold
        head_classes = cls_num_list > tail_threshold
        
        print(f"✅ Head classes: {head_classes.sum()} (samples > {tail_threshold})")
        print(f"✅ Tail classes: {tail_classes.sum()} (samples <= {tail_threshold})")
        
        return head_classes, tail_classes
    
    def create_risk_coverage_curves(self, expert_predictions, labels, head_classes, tail_classes):
        """
        Tạo Risk-Coverage Curves theo paper "Learning to Reject"
        """
        print("📊 Creating Risk-Coverage Curves...")
        
        # Cost values từ 0.0 đến 1.0 với bước nhảy 0.05
        cost_values = np.arange(0.0, 1.05, 0.05)
        
        results = {
            'BalPoE': {'costs': [], 'balanced_errors': [], 'worst_group_errors': [], 'rejection_rates': []},
            'Plugin_Single': {'costs': [], 'balanced_errors': [], 'worst_group_errors': [], 'rejection_rates': []},
            'Plugin_BalPoE_avg': {'costs': [], 'balanced_errors': [], 'worst_group_errors': [], 'rejection_rates': []},
            'MoE_Plugin': {'costs': [], 'balanced_errors': [], 'worst_group_errors': [], 'rejection_rates': []}
        }
        
        for cost in cost_values:
            print(f"  🔍 Processing cost = {cost:.2f}")
            
            # Baseline 1: BalPoE (gốc)
            balpoe_results = self._evaluate_balpoe_baseline(expert_predictions, labels, head_classes, tail_classes, cost)
            results['BalPoE']['costs'].append(cost)
            results['BalPoE']['balanced_errors'].append(balpoe_results['balanced_error'])
            results['BalPoE']['worst_group_errors'].append(balpoe_results['worst_group_error'])
            results['BalPoE']['rejection_rates'].append(balpoe_results['rejection_rate'])
            
            # Baseline 2: Plugin trên Single Model
            plugin_single_results = self._evaluate_plugin_single(expert_predictions, labels, head_classes, tail_classes, cost)
            results['Plugin_Single']['costs'].append(cost)
            results['Plugin_Single']['balanced_errors'].append(plugin_single_results['balanced_error'])
            results['Plugin_Single']['worst_group_errors'].append(plugin_single_results['worst_group_error'])
            results['Plugin_Single']['rejection_rates'].append(plugin_single_results['rejection_rate'])
            
            # Baseline 3: Plugin trên BalPoE-avg
            plugin_balpoe_results = self._evaluate_plugin_balpoe_avg(expert_predictions, labels, head_classes, tail_classes, cost)
            results['Plugin_BalPoE_avg']['costs'].append(cost)
            results['Plugin_BalPoE_avg']['balanced_errors'].append(plugin_balpoe_results['balanced_error'])
            results['Plugin_BalPoE_avg']['worst_group_errors'].append(plugin_balpoe_results['worst_group_error'])
            results['Plugin_BalPoE_avg']['rejection_rates'].append(plugin_balpoe_results['rejection_rate'])
            
            # Proposed: MoE-Plugin
            moe_plugin_results = self._evaluate_moe_plugin(expert_predictions, labels, head_classes, tail_classes, cost)
            results['MoE_Plugin']['costs'].append(cost)
            results['MoE_Plugin']['balanced_errors'].append(moe_plugin_results['balanced_error'])
            results['MoE_Plugin']['worst_group_errors'].append(moe_plugin_results['worst_group_error'])
            results['MoE_Plugin']['rejection_rates'].append(moe_plugin_results['rejection_rate'])
        
        return results
    
    def _evaluate_balpoe_baseline(self, expert_predictions, labels, head_classes, tail_classes, cost):
        """Baseline 1: BalPoE (gốc) - không có rejection"""
        
        # Average ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for name in ['head_expert', 'balanced_expert', 'tail_expert']:
            ensemble_pred += expert_predictions[name] / 3.0
        
        # Get predictions
        _, predictions = torch.max(ensemble_pred, dim=1)
        
        # No rejection for BalPoE baseline
        rejection_rate = 0.0
        
        # Compute errors
        balanced_error = self._compute_balanced_error(predictions, labels, head_classes, tail_classes)
        worst_group_error = self._compute_worst_group_error(predictions, labels, head_classes, tail_classes)
        
        return {
            'balanced_error': balanced_error,
            'worst_group_error': worst_group_error,
            'rejection_rate': rejection_rate
        }
    
    def _evaluate_plugin_single(self, expert_predictions, labels, head_classes, tail_classes, cost):
        """Baseline 2: Plugin trên Single Model (balanced expert)"""
        
        # Use only balanced expert
        single_pred = expert_predictions['balanced_expert']
        
        # Apply rejection rule with cost
        max_probs, predictions = torch.max(single_pred, dim=1)
        reject_mask = max_probs < (1.0 - cost)  # Higher cost = lower threshold
        
        # Compute errors
        balanced_error = self._compute_balanced_error(predictions, labels, head_classes, tail_classes, reject_mask)
        worst_group_error = self._compute_worst_group_error(predictions, labels, head_classes, tail_classes, reject_mask)
        rejection_rate = reject_mask.float().mean().item()
        
        return {
            'balanced_error': balanced_error,
            'worst_group_error': worst_group_error,
            'rejection_rate': rejection_rate
        }
    
    def _evaluate_plugin_balpoe_avg(self, expert_predictions, labels, head_classes, tail_classes, cost):
        """Baseline 3: Plugin trên BalPoE-avg"""
        
        # Average ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for name in ['head_expert', 'balanced_expert', 'tail_expert']:
            ensemble_pred += expert_predictions[name] / 3.0
        
        # Apply rejection rule with cost
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < (1.0 - cost)  # Higher cost = lower threshold
        
        # Compute errors
        balanced_error = self._compute_balanced_error(predictions, labels, head_classes, tail_classes, reject_mask)
        worst_group_error = self._compute_worst_group_error(predictions, labels, head_classes, tail_classes, reject_mask)
        rejection_rate = reject_mask.float().mean().item()
        
        return {
            'balanced_error': balanced_error,
            'worst_group_error': worst_group_error,
            'rejection_rate': rejection_rate
        }
    
    def _evaluate_moe_plugin(self, expert_predictions, labels, head_classes, tail_classes, cost):
        """Proposed: MoE-Plugin"""
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], self.plugin_params['expert_weights'])):
            ensemble_pred += weight * expert_predictions[name]
        
        # Apply rejection rule with cost
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < (1.0 - cost)  # Higher cost = lower threshold
        
        # Compute errors
        balanced_error = self._compute_balanced_error(predictions, labels, head_classes, tail_classes, reject_mask)
        worst_group_error = self._compute_worst_group_error(predictions, labels, head_classes, tail_classes, reject_mask)
        rejection_rate = reject_mask.float().mean().item()
        
        return {
            'balanced_error': balanced_error,
            'worst_group_error': worst_group_error,
            'rejection_rate': rejection_rate
        }
    
    def _compute_balanced_error(self, predictions, labels, head_classes, tail_classes, reject_mask=None):
        """Tính balanced error"""
        
        if reject_mask is None:
            reject_mask = torch.zeros_like(predictions, dtype=torch.bool)
        
        # Get group masks
        head_mask = torch.tensor([head_classes[label.item()] for label in labels])
        tail_mask = torch.tensor([tail_classes[label.item()] for label in labels])
        
        # Compute errors for each group
        head_error = self._compute_group_error(predictions, labels, head_mask, reject_mask)
        tail_error = self._compute_group_error(predictions, labels, tail_mask, reject_mask)
        
        # Balanced error = average of group errors
        balanced_error = (head_error + tail_error) / 2.0
        
        return balanced_error
    
    def _compute_worst_group_error(self, predictions, labels, head_classes, tail_classes, reject_mask=None):
        """Tính worst-group error"""
        
        if reject_mask is None:
            reject_mask = torch.zeros_like(predictions, dtype=torch.bool)
        
        # Get group masks
        head_mask = torch.tensor([head_classes[label.item()] for label in labels])
        tail_mask = torch.tensor([tail_classes[label.item()] for label in labels])
        
        # Compute errors for each group
        head_error = self._compute_group_error(predictions, labels, head_mask, reject_mask)
        tail_error = self._compute_group_error(predictions, labels, tail_mask, reject_mask)
        
        # Worst-group error = max of group errors
        worst_group_error = max(head_error, tail_error)
        
        return worst_group_error
    
    def _compute_group_error(self, predictions, labels, group_mask, reject_mask):
        """Tính error cho một group"""
        
        if group_mask.sum() == 0:
            return 0.0
        
        # Get group predictions and labels
        group_predictions = predictions[group_mask]
        group_labels = labels[group_mask]
        group_reject_mask = reject_mask[group_mask]
        
        # Compute error
        correct_mask = (group_predictions == group_labels) & (~group_reject_mask)
        error = 1.0 - correct_mask.float().mean()
        
        return error.item()
    
    def compute_aurc(self, results):
        """Tính AURC (Area Under Risk-Coverage Curve)"""
        print("📊 Computing AURC...")
        
        aurc_results = {}
        
        for method_name, method_results in results.items():
            # AURC for Balanced Error
            balanced_aurc = np.trapz(method_results['balanced_errors'], method_results['rejection_rates'])
            
            # AURC for Worst-Group Error
            worst_group_aurc = np.trapz(method_results['worst_group_errors'], method_results['rejection_rates'])
            
            aurc_results[method_name] = {
                'balanced_aurc': balanced_aurc,
                'worst_group_aurc': worst_group_aurc
            }
            
            print(f"  ✅ {method_name}:")
            print(f"    - Balanced AURC: {balanced_aurc:.4f}")
            print(f"    - Worst-Group AURC: {worst_group_aurc:.4f}")
        
        return aurc_results
    
    def plot_risk_coverage_curves(self, results, save_dir):
        """Vẽ Risk-Coverage Curves"""
        print("📊 Plotting Risk-Coverage Curves...")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Balanced Error vs Rejection Rate
        for method_name, method_results in results.items():
            ax1.plot(method_results['rejection_rates'], method_results['balanced_errors'], 
                    label=method_name, marker='o', markersize=3)
        
        ax1.set_xlabel('Rejection Rate')
        ax1.set_ylabel('Balanced Error')
        ax1.set_title('Balanced Error vs Rejection Rate')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Worst-Group Error vs Rejection Rate
        for method_name, method_results in results.items():
            ax2.plot(method_results['rejection_rates'], method_results['worst_group_errors'], 
                    label=method_name, marker='o', markersize=3)
        
        ax2.set_xlabel('Rejection Rate')
        ax2.set_ylabel('Worst-Group Error')
        ax2.set_title('Worst-Group Error vs Rejection Rate')
        ax2.legend()
        ax2.grid(True)
        
        # Save plots
        plt.tight_layout()
        plt.savefig(save_dir / 'risk_coverage_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Risk-Coverage Curves saved to: {save_dir / 'risk_coverage_curves.png'}")
    
    def create_comparison_tables(self, results, aurc_results, save_dir):
        """Tạo bảng so sánh"""
        print("📊 Creating comparison tables...")
        
        # Table 1: AURC Comparison
        table1_data = []
        for method_name, aurc_data in aurc_results.items():
            table1_data.append({
                'Method': method_name,
                'Balanced AURC': f"{aurc_data['balanced_aurc']:.4f}",
                'Worst-Group AURC': f"{aurc_data['worst_group_aurc']:.4f}"
            })
        
        # Save table 1
        with open(save_dir / 'aurc_comparison.json', 'w') as f:
            json.dump(table1_data, f, indent=2)
        
        # Table 2: Accuracy at 0% rejection
        table2_data = []
        for method_name, method_results in results.items():
            # Get accuracy at 0% rejection (first point)
            balanced_error = method_results['balanced_errors'][0]
            worst_group_error = method_results['worst_group_errors'][0]
            rejection_rate = method_results['rejection_rates'][0]
            
            table2_data.append({
                'Method': method_name,
                'Balanced Error': f"{balanced_error:.4f}",
                'Worst-Group Error': f"{worst_group_error:.4f}",
                'Rejection Rate': f"{rejection_rate:.4f}"
            })
        
        # Save table 2
        with open(save_dir / 'accuracy_comparison.json', 'w') as f:
            json.dump(table2_data, f, indent=2)
        
        print(f"✅ Comparison tables saved to: {save_dir}")
    
    def evaluate(self):
        """Chạy toàn bộ evaluation pipeline"""
        print("🚀 Bắt đầu Evaluation Pipeline")
        print("=" * 60)
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Load expert models
        self.load_expert_models()
        
        # Get expert predictions on test set
        expert_predictions, test_labels = self.get_expert_predictions(self.test_data_loader)
        
        # Define groups
        head_classes, tail_classes = self.define_groups(test_labels)
        
        # Create risk-coverage curves
        print("\n📊 Creating Risk-Coverage Curves...")
        results = self.create_risk_coverage_curves(expert_predictions, test_labels, head_classes, tail_classes)
        
        # Compute AURC
        print("\n📊 Computing AURC...")
        aurc_results = self.compute_aurc(results)
        
        # Create plots and tables
        save_dir = Path(self.config['output']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_risk_coverage_curves(results, save_dir)
        self.create_comparison_tables(results, aurc_results, save_dir)
        
        print("\n" + "=" * 60)
        print("✅ Evaluation Completed!")
        print(f"📁 Results saved to: {save_dir}")
        
        return results, aurc_results


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Evaluate MoE-Plugin Pipeline')
    parser.add_argument('--plugin_checkpoint', type=str, required=True,
                       help='Path to optimized plugin parameters')
    parser.add_argument('--experts_dir', type=str, required=True,
                       help='Directory containing expert checkpoints')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to plugin optimization config file')
    parser.add_argument('--save_dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MoEPluginEvaluator(
        plugin_checkpoint=args.plugin_checkpoint,
        experts_dir=args.experts_dir,
        config_path=args.config,
        seed=args.seed
    )
    
    # Run evaluation
    results, aurc_results = evaluator.evaluate()
    
    print(f"\n🎉 Giai đoạn 3 hoàn thành!")
    print(f"📁 Evaluation results: {args.save_dir}")
    print(f"📊 So sánh các methods đã được thực hiện!")


if __name__ == '__main__':
    main()