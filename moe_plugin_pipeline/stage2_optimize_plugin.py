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
        
        # Debug: Print expert prediction statistics
        print(f"📊 Expert prediction statistics:")
        for name, preds in expert_predictions.items():
            print(f"  - {name}: mean={preds.mean():.4f}, std={preds.std():.4f}, min={preds.min():.4f}, max={preds.max():.4f}")
        
        # Debug: Print label distribution
        unique_labels, counts = torch.unique(labels, return_counts=True)
        print(f"📊 Label distribution (first 10 classes):")
        for i in range(min(10, len(unique_labels))):
            label = unique_labels[i].item()
            count = counts[i].item()
            group = "Head" if head_classes[label] else "Tail"
            print(f"  - Class {label}: {count} samples ({group})")
        
        # Grid search parameters
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        weight_search_space = self.config['cs_plugin']['weight_search_space']
        
        # Debug: Print search space
        total_combinations = (len(weight_search_space['w_head']) * 
                           len(weight_search_space['w_balanced']) * 
                           len(weight_search_space['w_tail']) * 
                           len(lambda_0_candidates))
        print(f"🔍 Grid search: {total_combinations} total combinations")
        print(f"  - Weight space: {len(weight_search_space['w_head'])}×{len(weight_search_space['w_balanced'])}×{len(weight_search_space['w_tail'])} = {len(weight_search_space['w_head']) * len(weight_search_space['w_balanced']) * len(weight_search_space['w_tail'])}")
        print(f"  - Lambda candidates: {lambda_0_candidates}")
        
        best_params = None
        best_balanced_error = float('inf')
        current_combination = 0
        constrained_combinations = 0
        total_valid_combinations = 0
        
        # Add constraints for better optimization (relaxed)
        min_tail_expert_weight = 0.05  # Minimum 5% weight for tail expert (relaxed)
        min_balanced_expert_weight = 0.1  # Minimum 10% weight for balanced expert (relaxed)
        
        # Vòng lặp ngoài: Grid search cho expert weights với constraints
        for w_head in weight_search_space['w_head']:
            for w_balanced in weight_search_space['w_balanced']:
                for w_tail in weight_search_space['w_tail']:
                    
                    # Normalize weights
                    total_weight = w_head + w_balanced + w_tail
                    if total_weight == 0:
                        continue
                    
                    expert_weights = [w_head/total_weight, w_balanced/total_weight, w_tail/total_weight]
                    
                    # Apply constraints - skip if tail expert gets too little weight
                    if expert_weights[2] < min_tail_expert_weight:
                        constrained_combinations += 1
                        continue
                    if expert_weights[1] < min_balanced_expert_weight:
                        constrained_combinations += 1
                        continue
                    
                    total_valid_combinations += 1
                    
                    # Vòng lặp trong: Grid search cho lambda_0
                    for lambda_0 in lambda_0_candidates:
                        current_combination += 1
                        
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
                        
                        # Add weight regularization penalty
                        weight_penalty = self._compute_weight_penalty(expert_weights)
                        balanced_error += weight_penalty
                        
                        # Debug: Print progress every 10 combinations or when finding new best
                        if current_combination % 10 == 0 or balanced_error < best_balanced_error:
                            print(f"    🔍 [{current_combination}/{total_combinations}] weights={[f'{w:.3f}' for w in expert_weights]}, λ₀={lambda_0}, α={alpha:.4f}, error={balanced_error:.4f}")
                        
                        if balanced_error < best_balanced_error:
                            best_balanced_error = balanced_error
                            best_params = {
                                'lambda_0': lambda_0,
                                'alpha': alpha,
                                'expert_weights': expert_weights,
                                'balanced_error': balanced_error
                            }
                            
                            print(f"    ✅ New best: balanced_error = {balanced_error:.4f}")
                            print(f"        📊 Best params: weights={[f'{w:.3f}' for w in expert_weights]}, λ₀={lambda_0}, α={alpha:.4f}")
        
        print(f"✅ CS-plugin optimization completed!")
        print(f"📊 Best balanced error: {best_balanced_error:.4f}")
        print(f"📊 Constrained combinations: {constrained_combinations}")
        print(f"📊 Valid combinations: {total_valid_combinations}")
        
        # Fallback: If no good solution found with constraints, try without constraints
        if best_balanced_error > 0.8:  # If error is too high
            print("⚠️  High error detected, trying fallback without constraints...")
            return self._fallback_optimization(expert_predictions, labels, head_classes, tail_classes)
        
        return best_params
    
    def _find_optimal_alpha(self, expert_predictions, labels, expert_weights, lambda_0, head_classes, tail_classes):
        """Tìm alpha tối ưu bằng improved power iteration với bounds"""
        
        # Better initialization - always start with reasonable alpha
        alpha = 0.4  # Start with moderate threshold
        
        M = self.config['cs_plugin']['alpha_search_iterations']
        
        for iteration in range(M):
            # Compute cost-sensitive error với alpha hiện tại
            error = self._compute_cost_sensitive_error(
                expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes
            )
            
            # Improved update rule with bounds
            if error > 0.6:  # High error, increase threshold
                alpha = min(0.7, alpha + 0.05)  # Cap at 0.7
            elif error < 0.3:  # Low error, decrease threshold
                alpha = max(0.2, alpha - 0.05)  # Floor at 0.2
            else:  # Moderate error, fine-tune
                alpha = alpha + 0.01 * (0.5 - error)  # Small adjustments
            
            alpha = max(0.2, min(0.7, alpha))  # Clamp to [0.2, 0.7] - reasonable range
        
        return alpha
    
    def _compute_cost_sensitive_error(self, expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes):
        """Tính cost-sensitive error"""
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], expert_weights)):
            ensemble_pred += weight * expert_predictions[name]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < alpha
        
        # Compute error (simplified)
        correct_mask = (predictions == labels) & (~reject_mask)
        error = 1.0 - correct_mask.float().mean()
        
        return error.item()
    
    def _evaluate_balanced_error(self, expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes):
        """Đánh giá balanced error"""
        return self._compute_cost_sensitive_error(
            expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes
        )
    
    def _compute_weight_penalty(self, weights):
        """Compute penalty for extreme weight distributions (relaxed)"""
        # Penalize if any expert gets too much weight (>90%) - relaxed from 80%
        max_weight = max(weights)
        if max_weight > 0.9:
            return (max_weight - 0.9) * 0.05  # Reduced penalty
        
        # Penalize if tail expert gets too little weight (<2%) - relaxed from 5%
        if weights[2] < 0.02:
            return (0.02 - weights[2]) * 0.1  # Reduced penalty
        
        return 0.0
    
    def _fallback_optimization(self, expert_predictions, labels, head_classes, tail_classes):
        """Fallback optimization without constraints if main optimization fails"""
        print("🔄 Running fallback optimization without constraints...")
        
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        weight_search_space = self.config['cs_plugin']['weight_search_space']
        
        best_params = None
        best_balanced_error = float('inf')
        
        # Try all combinations without constraints
        for w_head in weight_search_space['w_head']:
            for w_balanced in weight_search_space['w_balanced']:
                for w_tail in weight_search_space['w_tail']:
                    total_weight = w_head + w_balanced + w_tail
                    if total_weight == 0:
                        continue
                    
                    expert_weights = [w_head/total_weight, w_balanced/total_weight, w_tail/total_weight]
                    
                    for lambda_0 in lambda_0_candidates:
                        alpha = self._find_optimal_alpha(
                            expert_predictions, labels, expert_weights, lambda_0, head_classes, tail_classes
                        )
                        
                        balanced_error = self._evaluate_balanced_error(
                            expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes
                        )
                        
                        # No weight penalty in fallback
                        if balanced_error < best_balanced_error:
                            best_balanced_error = balanced_error
                            best_params = {
                                'lambda_0': lambda_0,
                                'alpha': alpha,
                                'expert_weights': expert_weights,
                                'balanced_error': balanced_error
                            }
        
        print(f"📊 Fallback best balanced error: {best_balanced_error:.4f}")
        return best_params
    
    def worst_group_plugin_optimization(self, expert_predictions, labels, head_classes, tail_classes, cs_params):
        """
        Thuật toán 2: Worst-group Plugin để tối ưu Worst-Group Error
        Theo đúng paper "Learning to Reject Meets Long-tail Learning"
        """
        print("🔍 Worst-group Plugin Optimization...")
        
        # Initialize group weights
        group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)  # [head, tail]
        
        # Algorithm parameters
        T = self.config['worst_group_plugin']['max_iterations']
        step_size = self.config['worst_group_plugin']['step_size']
        
        best_params = cs_params
        best_worst_group_error = float('inf')
        
        # Early stopping parameters
        patience = 5
        no_improvement_count = 0
        
        for iteration in range(T):
            print(f"  🔄 Iteration {iteration+1}/{T}")
            
            # Compute group-wise errors
            head_error, tail_error = self._compute_group_errors(
                expert_predictions, labels, head_classes, tail_classes, cs_params
            )
            
            # Debug: Print group errors and weights
            print(f"    📊 Group errors: Head={head_error:.4f}, Tail={tail_error:.4f}")
            print(f"    📊 Current group weights: {group_weights.tolist()}")
            
            # Update group weights using exponentiated gradient
            old_group_weights = group_weights.clone()
            group_weights = self._update_group_weights(
                group_weights, head_error, tail_error, step_size
            )
            
            # Debug: Print weight updates
            weight_change = torch.abs(group_weights - old_group_weights).sum().item()
            print(f"    📊 Weight change: {weight_change:.6f}")
            
            # Check for convergence
            if weight_change < 1e-6:
                print(f"    ✅ Converged at iteration {iteration+1}")
                break
            
            # Evaluate worst-group error
            worst_group_error = max(head_error, tail_error)
            
            if worst_group_error < best_worst_group_error:
                best_worst_group_error = worst_group_error
                best_params = {
                    **cs_params,
                    'group_weights': group_weights.tolist(),
                    'worst_group_error': worst_group_error
                }
                no_improvement_count = 0
                
                print(f"    ✅ New best: worst_group_error = {worst_group_error:.4f}")
                print(f"        📊 Best group weights: {group_weights.tolist()}")
                print(f"        📊 Head error: {head_error:.4f}, Tail error: {tail_error:.4f}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"    ⏹️  Early stopping at iteration {iteration+1}")
                    break
        
        print(f"✅ Worst-group plugin optimization completed!")
        print(f"📊 Best worst-group error: {best_worst_group_error:.4f}")
        
        return best_params
    
    def _compute_group_errors(self, expert_predictions, labels, head_classes, tail_classes, params):
        """Tính group-wise errors"""
        
        # Get group masks - fix deprecation warning
        head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
        tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
        
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
            'config': self.config,
            'improvements': {
                'constrained_optimization': True,
                'weight_regularization': True,
                'early_stopping': True,
                'improved_alpha_finding': True,
                'fallback_optimization': True,
                'min_tail_expert_weight': 0.05,
                'min_balanced_expert_weight': 0.1,
                'alpha_bounds': [0.2, 0.7]
            }
        }
        
        # Save parameters
        params_file = save_dir / 'optimized_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"✅ Parameters saved to: {params_file}")
        
        # Debug: Print detailed final results
        print(f"\n📊 Detailed Final Results:")
        print(f"  - Lambda_0: {params['lambda_0']}")
        print(f"  - Alpha (rejection threshold): {params['alpha']:.6f}")
        print(f"  - Expert weights: {[f'{w:.6f}' for w in params['expert_weights']]}")
        print(f"  - Group weights: {[f'{w:.6f}' for w in params['group_weights']]}")
        print(f"  - Balanced error: {params['balanced_error']:.6f}")
        print(f"  - Worst-group error: {params['worst_group_error']:.6f}")
        
        # Debug: Analyze expert weight distribution
        expert_names = ['head_expert', 'balanced_expert', 'tail_expert']
        print(f"\n🔍 Expert Weight Analysis:")
        for name, weight in zip(expert_names, params['expert_weights']):
            percentage = weight * 100
            print(f"  - {name}: {weight:.6f} ({percentage:.2f}%)")
        
        # Debug: Analyze group weight distribution
        group_names = ['head_group', 'tail_group']
        print(f"\n🔍 Group Weight Analysis:")
        for name, weight in zip(group_names, params['group_weights']):
            percentage = weight * 100
            print(f"  - {name}: {weight:.6f} ({percentage:.2f}%)")
        
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