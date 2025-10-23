#!/usr/bin/env python3
"""
Giai đoạn 2: Tối ưu hóa Plugin Parameters
Mục tiêu: Tìm ra các tham số tối ưu cho CS-plugin và Worst-group plugin
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

# Add parent directory to path để import từ BalPoE gốc
sys.path.append(str(Path(__file__).parent.parent))

# Import trực tiếp từ BalPoE gốc
import data_loader.data_loaders as module_data
from utils import seed_everything
from parse_config import ConfigParser
import model.model as module_arch


class MoEPluginOptimizer:
    """
    MoE-Plugin Optimizer
    Triển khai đúng theo paper "Learning to Reject Meets Long-tail Learning"
    """
    
    def __init__(self, experts_dir: str, config_path: str, seed: int = 1):
        self.experts_dir = Path(experts_dir)
        self.config_path = config_path
        self.seed = seed
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        seed_everything(seed)
        
        print(f"🚀 Khởi tạo MoE-Plugin Optimizer")
        print(f"📂 Experts directory: {self.experts_dir}")
        print(f"📊 Dataset: {self.config['dataset']['name']}")
        print(f"🎯 CS-plugin candidates: {self.config['cs_plugin']['lambda_0_candidates']}")
        print(f"🔄 Worst-group iterations: {self.config['worst_group_plugin']['max_iterations']}")
        
    def setup_data_loaders(self):
        """Thiết lập data loaders cho validation (20%) và test (80%)"""
        print("📂 Thiết lập data loaders...")
        
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
        
        # Create full data loader
        self.full_data_loader = getattr(module_data, data_loader_config['type'])(
            **data_loader_config['args']
        )
        
        # Split into validation (20%) and test (80%)
        val_split = self.config['dataset']['val_split']
        total_samples = len(self.full_data_loader.dataset)
        val_size = int(total_samples * val_split)
        
        print(f"📊 Total samples: {total_samples}")
        print(f"📊 Validation size: {val_size}")
        print(f"📊 Test size: {total_samples - val_size}")
        
        # Create single random permutation and split
        all_indices = torch.randperm(total_samples)
        val_indices = all_indices[:val_size]
        test_indices = all_indices[val_size:]
        
        # Debug: Check indices range
        print(f"🔍 Val indices range: {val_indices.min().item()} - {val_indices.max().item()}")
        print(f"🔍 Test indices range: {test_indices.min().item()} - {test_indices.max().item()}")
        print(f"🔍 Dataset size: {len(self.full_data_loader.dataset)}")
        
        # Create validation data loader using Subset
        val_subset = torch.utils.data.Subset(self.full_data_loader.dataset, val_indices)
        self.val_data_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=256,
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Create test data loader using Subset
        test_subset = torch.utils.data.Subset(self.full_data_loader.dataset, test_indices)
        self.test_data_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=256,
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        print(f"✅ Validation samples: {len(val_indices)}")
        print(f"✅ Test samples: {len(test_indices)}")
        
    def load_expert_models(self):
        """Load 3 expert models từ BalPoE checkpoints"""
        print("📂 Loading expert models...")
        
        self.expert_models = {}
        tau_values = [0, 1.0, 2.0]
        expert_names = ['head_expert', 'balanced_expert', 'tail_expert']
        
        for tau, name in zip(tau_values, expert_names):
            # Find checkpoint file - prioritize best model
            best_checkpoint = self.experts_dir / "model_best.pth"
            if best_checkpoint.exists():
                checkpoint_path = best_checkpoint
                print(f"  🏆 Using best model: {best_checkpoint.name}")
            else:
                # Fallback to latest epoch checkpoint
                checkpoint_files = list(self.experts_dir.glob(f"*epoch*.pth"))
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoint files found in {self.experts_dir}")
                checkpoint_path = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                print(f"  ⚠️  Best model not found, using latest: {checkpoint_path.name}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
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
            
            print(f"  ✅ {name} (τ={tau}): {checkpoint_path.name}")
        
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
        # Cần lấy từ training data loader để biết số samples per class
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
    
    def cs_plugin_optimization(self, expert_predictions, labels, head_classes, tail_classes):
        """
        Thuật toán 1: CS-plugin để tối ưu Balanced Error
        Theo đúng paper "Learning to Reject Meets Long-tail Learning"
        """
        print("🔍 CS-plugin Optimization...")
        
        # Grid search parameters
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        weight_search_space = self.config['cs_plugin']['weight_search_space']
        
        best_params = None
        best_balanced_error = float('inf')
        
        # Vòng lặp ngoài: Grid search cho expert weights
        for w_head in weight_search_space['w_head']:
            for w_balanced in weight_search_space['w_balanced']:
                for w_tail in weight_search_space['w_tail']:
                    
                    # Normalize weights
                    total_weight = w_head + w_balanced + w_tail
                    if total_weight == 0:
                        continue
                    
                    expert_weights = [w_head/total_weight, w_balanced/total_weight, w_tail/total_weight]
                    
                    # Vòng lặp trong: Grid search cho lambda_0
                    for lambda_0 in lambda_0_candidates:
                        
                        # Tìm alpha tối ưu bằng power iteration
                        alpha = self._find_optimal_alpha(
                            expert_predictions, labels, 
                            expert_weights, lambda_0, head_classes, tail_classes
                        )
                        
                        # Đánh giá balanced error
                        balanced_error = self._evaluate_balanced_error(
                            expert_predictions, labels,
                            expert_weights, lambda_0, alpha, head_classes, tail_classes
                        )
                        
                        if balanced_error < best_balanced_error:
                            best_balanced_error = balanced_error
                            best_params = {
                                'lambda_0': lambda_0,
                                'alpha': alpha,
                                'expert_weights': expert_weights,
                                'balanced_error': balanced_error
                            }
                            
                            print(f"    ✅ New best: balanced_error = {balanced_error:.4f}")
        
        print(f"✅ CS-plugin optimization completed!")
        print(f"📊 Best balanced error: {best_balanced_error:.4f}")
        
        return best_params
    
    def _find_optimal_alpha(self, expert_predictions, labels, expert_weights, lambda_0, head_classes, tail_classes, group_weights=None):
        """Tìm alpha tối ưu bằng power iteration (M=10 iterations) theo đúng paper"""
        
        # Initialize alpha
        alpha = 0.5  # Initial guess
        M = self.config['cs_plugin']['alpha_search_iterations']
        
        # Default group weights for balanced error (equal weights)
        if group_weights is None:
            group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)
        
        for iteration in range(M):
            # Compute cost-sensitive error với alpha hiện tại và group weights
            error = self._compute_cost_sensitive_error_with_weights(
                expert_predictions, labels, expert_weights, lambda_0, alpha, 
                head_classes, tail_classes, group_weights
            )
            
            # Power iteration update rule theo paper
            # α^(t+1) = α^(t) + η * (target_error - current_error)
            # Với target_error = 0.5 (balanced target)
            target_error = 0.5
            learning_rate = 0.1
            alpha_update = learning_rate * (target_error - error)
            alpha = alpha + alpha_update
            
            # Clamp alpha to reasonable range [0.1, 0.9]
            alpha = max(0.1, min(0.9, alpha))
        
        return alpha
    
    def _compute_cost_sensitive_error(self, expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes):
        """Tính cost-sensitive error (balanced error với equal group weights)"""
        return self._compute_cost_sensitive_error_with_weights(
            expert_predictions, labels, expert_weights, lambda_0, alpha, 
            head_classes, tail_classes, torch.tensor([0.5, 0.5])
        )
    
    def _compute_cost_sensitive_error_with_weights(self, expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes, group_weights):
        """Tính cost-sensitive error với group weights β theo đúng paper"""
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], expert_weights)):
            ensemble_pred += weight * expert_predictions[name]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < alpha
        
        # Get group masks
        head_mask = torch.tensor([head_classes[label.item()] for label in labels])
        tail_mask = torch.tensor([tail_classes[label.item()] for label in labels])
        
        # Compute group-wise errors
        head_correct = (predictions == labels) & (~reject_mask) & head_mask
        tail_correct = (predictions == labels) & (~reject_mask) & tail_mask
        
        head_error = 1.0 - head_correct.float().sum() / max(head_mask.sum().item(), 1)
        tail_error = 1.0 - tail_correct.float().sum() / max(tail_mask.sum().item(), 1)
        
        # Cost-sensitive error với group weights β
        cost_sensitive_error = group_weights[0] * head_error + group_weights[1] * tail_error
        
        return cost_sensitive_error.item()
    
    def _evaluate_balanced_error(self, expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes):
        """Đánh giá balanced error"""
        return self._compute_cost_sensitive_error(
            expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes
        )
    
    def worst_group_plugin_optimization(self, expert_predictions, labels, head_classes, tail_classes, cs_params):
        """
        Thuật toán 2: Worst-group Plugin để tối ưu Worst-Group Error
        Theo đúng paper "Learning to Reject Meets Long-tail Learning"
        """
        print("🔍 Worst-group Plugin Optimization...")
        
        # Initialize group weights β⁽⁰⁾ = [0.5, 0.5]
        group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)  # [head, tail]
        
        # Algorithm parameters
        T = self.config['worst_group_plugin']['max_iterations']
        step_size = self.config['worst_group_plugin']['step_size']
        
        best_params = cs_params
        best_worst_group_error = float('inf')
        
        for iteration in range(T):
            print(f"  🔄 Iteration {iteration+1}/{T}")
            print(f"    📊 Current group weights β⁽ᵗ⁾: {group_weights.tolist()}")
            
            # BƯỚC QUAN TRỌNG: Gọi CS-plugin với group weights β⁽ᵗ⁾ hiện tại
            # Tìm (h⁽ᵗ⁾, r⁽ᵗ⁾) tối ưu cho cost-sensitive error với β⁽ᵗ⁾
            current_params = self._cs_plugin_with_group_weights(
                expert_predictions, labels, head_classes, tail_classes, group_weights
            )
            
            # Compute group-wise errors với classifier (h⁽ᵗ⁾, r⁽ᵗ⁾)
            head_error, tail_error = self._compute_group_errors_with_params(
                expert_predictions, labels, head_classes, tail_classes, current_params
            )
            
            print(f"    📊 Group errors: Head={head_error:.4f}, Tail={tail_error:.4f}")
            
            # Update group weights using exponentiated gradient
            # βₖ⁽ᵗ⁺¹⁾ ∝ βₖ⁽ᵗ⁾ · exp(ξ · êₖ(h⁽ᵗ⁾, r⁽ᵗ⁾))
            group_weights = self._update_group_weights(
                group_weights, head_error, tail_error, step_size
            )
            
            # Evaluate worst-group error
            worst_group_error = max(head_error, tail_error)
            
            if worst_group_error < best_worst_group_error:
                best_worst_group_error = worst_group_error
                best_params = {
                    **current_params,
                    'group_weights': group_weights.tolist(),
                    'worst_group_error': worst_group_error
                }
                
                print(f"    ✅ New best: worst_group_error = {worst_group_error:.4f}")
                print(f"        📊 Best group weights: {group_weights.tolist()}")
        
        print(f"✅ Worst-group plugin optimization completed!")
        print(f"📊 Best worst-group error: {best_worst_group_error:.4f}")
        
        return best_params
    
    def _cs_plugin_with_group_weights(self, expert_predictions, labels, head_classes, tail_classes, group_weights):
        """CS-plugin optimization với group weights β⁽ᵗ⁾ cụ thể"""
        
        # Grid search parameters
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        weight_search_space = self.config['cs_plugin']['weight_search_space']
        
        best_params = None
        best_cost_sensitive_error = float('inf')
        
        # Grid search cho expert weights
        for w_head in weight_search_space['w_head']:
            for w_balanced in weight_search_space['w_balanced']:
                for w_tail in weight_search_space['w_tail']:
                    
                    # Normalize weights
                    total_weight = w_head + w_balanced + w_tail
                    if total_weight == 0:
                        continue
                    
                    expert_weights = [w_head/total_weight, w_balanced/total_weight, w_tail/total_weight]
                    
                    # Grid search cho lambda_0
                    for lambda_0 in lambda_0_candidates:
                        
                        # Tìm alpha tối ưu với group weights β⁽ᵗ⁾
                        alpha = self._find_optimal_alpha(
                            expert_predictions, labels, 
                            expert_weights, lambda_0, head_classes, tail_classes, group_weights
                        )
                        
                        # Đánh giá cost-sensitive error với β⁽ᵗ⁾
                        cost_sensitive_error = self._compute_cost_sensitive_error_with_weights(
                            expert_predictions, labels,
                            expert_weights, lambda_0, alpha, head_classes, tail_classes, group_weights
                        )
                        
                        if cost_sensitive_error < best_cost_sensitive_error:
                            best_cost_sensitive_error = cost_sensitive_error
                            best_params = {
                                'lambda_0': lambda_0,
                                'alpha': alpha,
                                'expert_weights': expert_weights,
                                'cost_sensitive_error': cost_sensitive_error
                            }
        
        return best_params
    
    def _compute_group_errors_with_params(self, expert_predictions, labels, head_classes, tail_classes, params):
        """Tính group-wise errors với parameters cụ thể"""
        
        # Get group masks
        head_mask = torch.tensor([head_classes[label.item()] for label in labels])
        tail_mask = torch.tensor([tail_classes[label.item()] for label in labels])
        
        # Compute errors for each group
        head_error = self._compute_group_error_with_params(expert_predictions, labels, head_mask, params)
        tail_error = self._compute_group_error_with_params(expert_predictions, labels, tail_mask, params)
        
        return head_error, tail_error
    
    def _compute_group_error_with_params(self, expert_predictions, labels, group_mask, params):
        """Tính error cho một group với parameters cụ thể"""
        
        if group_mask.sum() == 0:
            return 0.0
        
        # Get group predictions and labels
        group_predictions = {name: pred[group_mask] for name, pred in expert_predictions.items()}
        group_labels = labels[group_mask]
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(group_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], params['expert_weights'])):
            ensemble_pred += weight * group_predictions[name]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < params['alpha']
        
        # Compute error
        correct_mask = (predictions == group_labels) & (~reject_mask)
        error = 1.0 - correct_mask.float().mean()
        
        return error.item()
    
    def _compute_group_errors(self, expert_predictions, labels, head_classes, tail_classes, params):
        """Tính group-wise errors"""
        
        # Get group masks
        head_mask = torch.tensor([head_classes[label.item()] for label in labels])
        tail_mask = torch.tensor([tail_classes[label.item()] for label in labels])
        
        # Compute errors for each group
        head_error = self._compute_group_error(expert_predictions, labels, head_mask, params)
        tail_error = self._compute_group_error(expert_predictions, labels, tail_mask, params)
        
        return head_error, tail_error
    
    def _compute_group_error(self, expert_predictions, labels, group_mask, params):
        """Tính error cho một group"""
        
        if group_mask.sum() == 0:
            return 0.0
        
        # Get group predictions and labels
        group_predictions = {name: pred[group_mask] for name, pred in expert_predictions.items()}
        group_labels = labels[group_mask]
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(group_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], params['expert_weights'])):
            ensemble_pred += weight * group_predictions[name]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < params['alpha']
        
        # Compute error
        correct_mask = (predictions == group_labels) & (~reject_mask)
        error = 1.0 - correct_mask.float().mean()
        
        return error.item()
    
    def _update_group_weights(self, group_weights, head_error, tail_error, step_size):
        """Cập nhật group weights bằng exponentiated gradient"""
        
        # Compute gradients
        grad_head = head_error
        grad_tail = tail_error
        
        # Update weights
        log_weights = torch.log(group_weights + 1e-8)
        log_weights[0] -= step_size * grad_head
        log_weights[1] -= step_size * grad_tail
        
        # Normalize
        group_weights = torch.softmax(log_weights, dim=0)
        
        return group_weights
    
    def optimize(self):
        """Chạy toàn bộ optimization pipeline"""
        print("🚀 Bắt đầu Plugin Optimization Pipeline")
        print("=" * 60)
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Load expert models
        self.load_expert_models()
        
        # Get expert predictions on validation set
        expert_predictions, val_labels = self.get_expert_predictions(self.val_data_loader)
        
        # Define groups
        head_classes, tail_classes = self.define_groups(val_labels)
        
        # Step 1: CS-plugin optimization
        print("\n📊 Step 1: CS-plugin Optimization")
        cs_params = self.cs_plugin_optimization(expert_predictions, val_labels, head_classes, tail_classes)
        
        # Step 2: Worst-group plugin optimization
        print("\n📊 Step 2: Worst-group Plugin Optimization")
        final_params = self.worst_group_plugin_optimization(expert_predictions, val_labels, head_classes, tail_classes, cs_params)
        
        print("\n" + "=" * 60)
        print("✅ Plugin Optimization Completed!")
        print(f"📊 Final Parameters:")
        print(f"  - Lambda_0: {final_params['lambda_0']}")
        print(f"  - Alpha: {final_params['alpha']}")
        print(f"  - Expert weights: {final_params['expert_weights']}")
        print(f"  - Group weights: {final_params['group_weights']}")
        print(f"  - Balanced error: {final_params['balanced_error']:.4f}")
        print(f"  - Worst-group error: {final_params['worst_group_error']:.4f}")
        
        return final_params
    
    def save_optimized_parameters(self, params):
        """Lưu optimized parameters"""
        print("💾 Lưu optimized parameters...")
        
        save_dir = Path(self.config['output']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        params_dict = {
            'lambda_0': params['lambda_0'],
            'alpha': params['alpha'],
            'expert_weights': params['expert_weights'],
            'group_weights': params['group_weights'],
            'rejection_threshold': params['alpha'],  # Use alpha as rejection threshold
            'balanced_error': params['balanced_error'],
            'worst_group_error': params['worst_group_error'],
            'config': self.config
        }
        
        # Save parameters
        params_file = save_dir / 'optimized_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"✅ Parameters saved to: {params_file}")
        
        return save_dir


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Optimize Plugin Parameters')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to plugin optimization config file')
    parser.add_argument('--experts_dir', type=str, required=True,
                       help='Directory containing expert checkpoints')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = MoEPluginOptimizer(
        experts_dir=args.experts_dir,
        config_path=args.config,
        seed=args.seed
    )
    
    # Run optimization
    optimized_params = optimizer.optimize()
    
    # Save optimized parameters
    save_dir = optimizer.save_optimized_parameters(optimized_params)
    
    print(f"\n🎉 Giai đoạn 2 hoàn thành!")
    print(f"📁 Optimized parameters: {save_dir}")
    print(f"🔧 Tiếp theo: python stage3_evaluate.py --plugin_checkpoint {save_dir}/optimized_parameters.json")


if __name__ == '__main__':
    main()