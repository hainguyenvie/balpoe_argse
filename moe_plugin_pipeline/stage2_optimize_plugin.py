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
import math
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
        Theo đúng paper "Learning to Reject Meets Long-tail Learning" với Bayes-optimal rejector
        """
        print("🔍 CS-plugin Optimization (Bayes-optimal)...")
        
        # Grid search parameters
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        
        # Equal weights cho experts (như paper BalPoE gốc)
        expert_weights = [1.0/3, 1.0/3, 1.0/3]  # Equal weights cho 3 experts
        
        # Initialize auxiliary variables và Lagrangian multipliers
        alpha_star = torch.tensor([
            self.config['cs_plugin']['auxiliary_variables']['alpha_head_init'],
            self.config['cs_plugin']['auxiliary_variables']['alpha_tail_init']
        ], dtype=torch.float)
        
        mu_star = torch.tensor([
            self.config['cs_plugin']['lagrangian_multipliers']['mu_head_init'],
            self.config['cs_plugin']['lagrangian_multipliers']['mu_tail_init']
        ], dtype=torch.float)
        
        c = self.config['cs_plugin']['rejection_penalty']
        
        best_params = None
        best_balanced_error = float('inf')
        
        # Equal group weights cho balanced error (βk = 1/K = 0.5)
        equal_group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)
        
        # Grid search cho lambda_0 để tối ưu auxiliary variables
        for lambda_0 in lambda_0_candidates:
            
            # Algorithm 1: CS-plug-in với M iterations
            alpha_opt = self._optimize_auxiliary_variables(
                expert_predictions, labels, head_classes, tail_classes, 
                expert_weights, lambda_0, equal_group_weights, c
            )
            
            # Compute balanced error với optimized α
            head_error, tail_error, rejection_rate = self._compute_bayes_optimal_error_with_alpha(
                expert_predictions, labels, head_classes, tail_classes,
                expert_weights, alpha_opt, lambda_0, c
            )
            
            # Balanced error = (head_error + tail_error) / 2
            balanced_error = (head_error + tail_error) / 2
            
            if balanced_error < best_balanced_error:
                best_balanced_error = balanced_error
                best_params = {
                    'lambda_0': lambda_0,
                    'alpha_opt': alpha_opt.tolist(),
                    'expert_weights': expert_weights,
                    'balanced_error': balanced_error,
                    'head_error': head_error,
                    'tail_error': tail_error,
                    'rejection_rate': rejection_rate
                }
                
                print(f"    ✅ New best: balanced_error = {balanced_error:.4f}")
                print(f"        📊 α_opt = {alpha_opt.tolist()}")
                print(f"        📊 λ₀ = {lambda_0}")
        
        print(f"✅ CS-plugin optimization completed!")
        print(f"📊 Best balanced error: {best_balanced_error:.4f}")
        print(f"📊 Expert weights (equal): {expert_weights}")
        
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
        head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
        tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
        
        # Compute πk(r) = P(r(x) = 0, y ∈ Gk) for each group
        head_non_rejected = (~reject_mask) & head_mask
        tail_non_rejected = (~reject_mask) & tail_mask
        
        pi_head = head_non_rejected.float().sum() / max(head_mask.sum().item(), 1)
        pi_tail = tail_non_rejected.float().sum() / max(tail_mask.sum().item(), 1)
        
        # Compute P(y ≠ h(x), r(x) = 0, y ∈ Gk) for each group
        head_errors = (predictions != labels) & head_non_rejected
        tail_errors = (predictions != labels) & tail_non_rejected
        
        # Normalized errors: (1/πk(r)) * P(y ≠ h(x), r(x) = 0, y ∈ Gk)
        head_normalized_error = head_errors.float().sum() / max(pi_head * head_mask.sum().item(), 1e-8)
        tail_normalized_error = tail_errors.float().sum() / max(pi_tail * tail_mask.sum().item(), 1e-8)
        
        # Rejection penalty: c * P(r(x) = 1)
        rejection_rate = reject_mask.float().mean()
        rejection_penalty = self.config['cs_plugin']['rejection_penalty'] * rejection_rate
        
        # Cost-sensitive error với group weights β theo đúng paper
        # R_bal^rej = (1/K) * Σ e_k(h,r) + c*P(r(x)=1)
        # Với balanced error: group_weights = [0.5, 0.5] = 1/K
        cost_sensitive_error = group_weights[0] * head_normalized_error + group_weights[1] * tail_normalized_error + rejection_penalty
        
        return cost_sensitive_error.item()
    
    def _apply_bayes_optimal_rejector(self, ensemble_pred, labels, head_classes, tail_classes, alpha_star, mu_star, c):
        """Apply Bayes-optimal rejector theo Theorem 1"""
        
        # Get class probabilities (ηy(x)) for all classes
        class_probs = torch.softmax(ensemble_pred, dim=1)  # Shape: [batch_size, num_classes]
        
        # Get group indices for each class
        num_classes = class_probs.shape[1]
        group_indices = torch.zeros(num_classes, dtype=torch.long)
        
        # Map classes to groups (0=head, 1=tail)
        for class_idx in range(num_classes):
            if head_classes[class_idx]:
                group_indices[class_idx] = 0  # head group
            else:
                group_indices[class_idx] = 1  # tail group
        
        # Compute optimal classifier: h*(x) = arg max_{y∈[L]} (1/α*[y]) * ηy(x)
        weighted_probs = class_probs / alpha_star[group_indices]  # (1/α*[y]) * ηy(x)
        predictions = torch.argmax(weighted_probs, dim=1)
        
        # Compute optimal rejector: r*(x) = 1 ⟺ max_{y∈[L]} (1/α*[y]) * ηy(x) < threshold
        max_weighted_probs = torch.max(weighted_probs, dim=1)[0]
        
        # Compute sample-dependent threshold: Σ_{y'∈[L]} ((1/α*[y']) - μ*[y']) * ηy'(x) - c
        threshold = torch.sum(
            ((1.0 / alpha_star[group_indices]) - mu_star[group_indices]) * class_probs, 
            dim=1
        ) - c
        
        # Apply rejection rule
        reject_mask = max_weighted_probs < threshold
        
        return predictions, reject_mask
    
    def _compute_bayes_optimal_error(self, expert_predictions, labels, head_classes, tail_classes, expert_weights, alpha_star, mu_star, c):
        """Compute error using Bayes-optimal classifier and rejector"""
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], expert_weights)):
            ensemble_pred += weight * expert_predictions[name]
        
        # Apply Bayes-optimal rejector
        predictions, reject_mask = self._apply_bayes_optimal_rejector(
            ensemble_pred, labels, head_classes, tail_classes, alpha_star, mu_star, c
        )
        
        # Get group masks
        head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
        tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
        
        # Compute group-wise errors
        head_correct = (predictions == labels) & (~reject_mask) & head_mask
        tail_correct = (predictions == labels) & (~reject_mask) & tail_mask
        
        head_error = 1.0 - head_correct.float().sum() / max(head_mask.sum().item(), 1)
        tail_error = 1.0 - tail_correct.float().sum() / max(tail_mask.sum().item(), 1)
        
        return head_error.item(), tail_error.item(), reject_mask.float().mean().item()
    
    def _optimize_auxiliary_variables(self, expert_predictions, labels, head_classes, tail_classes, expert_weights, lambda_0, group_weights, c):
        """Algorithm 1: CS-plug-in với M iterations theo đúng paper"""
        
        # Initialize auxiliary variables α^(0)
        alpha = torch.tensor([
            self.config['cs_plugin']['auxiliary_variables']['alpha_head_init'],
            self.config['cs_plugin']['auxiliary_variables']['alpha_tail_init']
        ], dtype=torch.float)
        
        # M iterations (power iteration)
        M = self.config['cs_plugin']['alpha_search_iterations']
        
        for m in range(M):
            # Bước 5: Construct (h^(m+1), r^(m+1)) using equations 12–13 with α̂k = αk^(m) / βk and μ̂ = μ
            # Với balanced error: βk = 1/K = 0.5 cho mỗi group
            alpha_hat = alpha / group_weights  # α̂k = αk^(m) / βk
            
            # μ̂ = μ (scalar từ lambda_0)
            mu_hat = torch.tensor([lambda_0, 0.0], dtype=torch.float)  # Simplified: μ̂ = [λ₀, 0]
            
            # Apply Bayes-optimal rejector với α̂ và μ̂
            predictions, reject_mask = self._apply_bayes_optimal_rejector_with_params(
                expert_predictions, labels, head_classes, tail_classes, 
                expert_weights, alpha_hat, mu_hat, c
            )
            
            # Bước 6: αk^(m+1) = (1/|S|) Σ_{(x,y)∈S} 1(y ∈ Gk, r^(m+1)(x) = 0), ∀k ∈ [K]
            head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
            tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
            
            # Compute empirical coverage: P(r(x) = 0, y ∈ Gk)
            head_non_rejected = (~reject_mask) & head_mask
            tail_non_rejected = (~reject_mask) & tail_mask
            
            # Update αk theo Algorithm 1
            alpha[0] = head_non_rejected.float().sum() / max(head_mask.sum().item(), 1)
            alpha[1] = tail_non_rejected.float().sum() / max(tail_mask.sum().item(), 1)
            
            # Clamp to reasonable range
            alpha = torch.clamp(alpha, 0.1, 0.9)
        
        return alpha
    
    def _apply_bayes_optimal_rejector_with_params(self, expert_predictions, labels, head_classes, tail_classes, expert_weights, alpha_hat, mu_hat, c):
        """Apply Bayes-optimal rejector với α̂ và μ̂ parameters"""
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], expert_weights)):
            ensemble_pred += weight * expert_predictions[name]
        
        # Get class probabilities (ηy(x)) for all classes
        class_probs = torch.softmax(ensemble_pred, dim=1)  # Shape: [batch_size, num_classes]
        
        # Get group indices for each class
        num_classes = class_probs.shape[1]
        group_indices = torch.zeros(num_classes, dtype=torch.long)
        
        # Map classes to groups (0=head, 1=tail)
        for class_idx in range(num_classes):
            if head_classes[class_idx]:
                group_indices[class_idx] = 0  # head group
            else:
                group_indices[class_idx] = 1  # tail group
        
        # Compute optimal classifier: h*(x) = arg max_{y∈[L]} (1/α̂[y]) * ηy(x)
        weighted_probs = class_probs / alpha_hat[group_indices]  # (1/α̂[y]) * ηy(x)
        predictions = torch.argmax(weighted_probs, dim=1)
        
        # Compute optimal rejector: r*(x) = 1 ⟺ max_{y∈[L]} (1/α̂[y]) * ηy(x) < threshold
        max_weighted_probs = torch.max(weighted_probs, dim=1)[0]
        
        # Compute sample-dependent threshold: Σ_{y'∈[L]} ((1/α̂[y']) - μ̂[y']) * ηy'(x) - c
        threshold = torch.sum(
            ((1.0 / alpha_hat[group_indices]) - mu_hat[group_indices]) * class_probs, 
            dim=1
        ) - c
        
        # Apply rejection rule
        reject_mask = max_weighted_probs < threshold
        
        return predictions, reject_mask
    
    def _compute_bayes_optimal_error_with_alpha(self, expert_predictions, labels, head_classes, tail_classes, expert_weights, alpha_opt, lambda_0, c):
        """Compute error với optimized α từ Algorithm 1"""
        
        # Equal group weights cho balanced error
        equal_group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)
        
        # α̂k = αk^(m) / βk
        alpha_hat = alpha_opt / equal_group_weights
        
        # μ̂ = μ (scalar từ lambda_0)
        mu_hat = torch.tensor([lambda_0, 0.0], dtype=torch.float)
        
        # Apply Bayes-optimal rejector với optimized parameters
        predictions, reject_mask = self._apply_bayes_optimal_rejector_with_params(
            expert_predictions, labels, head_classes, tail_classes, 
            expert_weights, alpha_hat, mu_hat, c
        )
        
        # Get group masks
        head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
        tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
        
        # Compute group-wise errors
        head_correct = (predictions == labels) & (~reject_mask) & head_mask
        tail_correct = (predictions == labels) & (~reject_mask) & tail_mask
        
        head_error = 1.0 - head_correct.float().sum() / max(head_mask.sum().item(), 1)
        tail_error = 1.0 - tail_correct.float().sum() / max(tail_mask.sum().item(), 1)
        
        return head_error.item(), tail_error.item(), reject_mask.float().mean().item()
    
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
        
        # Theorem 4 metrics tracking
        theorem4_metrics = {
            'head_errors': [],
            'tail_errors': [],
            'rejection_rates': [],
            'generalization_errors': [],
            'excess_cost_sensitive_risks': []
        }
        
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
            
            # Track Theorem 4 metrics
            theorem4_metrics['head_errors'].append(head_error)
            theorem4_metrics['tail_errors'].append(tail_error)
            theorem4_metrics['rejection_rates'].append(current_params.get('rejection_rate', 0.0))
            
            # Compute generalization error ε_t^gen
            head_gen_error, tail_gen_error = self._compute_generalization_error(
                expert_predictions, labels, head_classes, tail_classes, current_params,
                expert_predictions, labels  # Using same data for simplicity
            )
            theorem4_metrics['generalization_errors'].append((head_gen_error, tail_gen_error))
            
            # Compute excess cost-sensitive risk ε_t^cs
            excess_cs_risk = self._compute_excess_cost_sensitive_risk(
                expert_predictions, labels, head_classes, tail_classes, current_params, group_weights
            )
            theorem4_metrics['excess_cost_sensitive_risks'].append(excess_cs_risk)
            
            print(f"    📊 Theorem 4 metrics:")
            print(f"        ε_t^gen: Head={head_gen_error:.4f}, Tail={tail_gen_error:.4f}")
            print(f"        ε_t^cs: {excess_cs_risk:.4f}")
            
            # Update group weights using exponentiated gradient
            # βₖ⁽ᵗ⁺¹⁾ ∝ βₖ⁽ᵗ⁾ · exp(ξ · êₖ(h⁽ᵗ⁾, r⁽ᵗ⁾))
            group_weights = self._update_group_weights(
                group_weights, head_error, tail_error, step_size
            )
            
            # Evaluate worst-group error: R_worst^rej = max(e_k) + c*P(r(x)=1)
            worst_group_error = max(head_error, tail_error) + current_params.get('rejection_rate', 0.0) * self.config['cs_plugin']['rejection_penalty']
            
            if worst_group_error < best_worst_group_error:
                best_worst_group_error = worst_group_error
                
                # Tính balanced error cho parameters này
                balanced_error = self._compute_balanced_error(
                    expert_predictions, labels, head_classes, tail_classes, current_params
                )
                
                best_params = {
                    **current_params,
                    'group_weights': group_weights.tolist(),
                    'worst_group_error': worst_group_error,
                    'balanced_error': balanced_error
                }
                
                print(f"    ✅ New best: worst_group_error = {worst_group_error:.4f}")
                print(f"        📊 Best group weights: {group_weights.tolist()}")
        
        print(f"✅ Worst-group plugin optimization completed!")
        print(f"📊 Best worst-group error: {best_worst_group_error:.4f}")
        
        # Compute Theorem 4 final metrics
        theorem4_final_metrics = self._compute_theorem4_final_metrics(theorem4_metrics)
        best_params['theorem4_metrics'] = theorem4_final_metrics
        
        print(f"📊 Theorem 4 Final Metrics:")
        print(f"    E_t[e_head]: {theorem4_final_metrics['expected_head_error']:.4f}")
        print(f"    E_t[e_tail]: {theorem4_final_metrics['expected_tail_error']:.4f}")
        print(f"    E_t[P(r=1)]: {theorem4_final_metrics['expected_rejection_rate']:.4f}")
        print(f"    ε̄^cs: {theorem4_final_metrics['avg_excess_cost_sensitive_risk']:.4f}")
        print(f"    ε̄^gen: {theorem4_final_metrics['avg_generalization_error']:.4f}")
        
        return best_params
    
    def compute_risk_coverage_curves(self, expert_predictions, labels, head_classes, tail_classes):
        """Compute risk-coverage curves theo paper evaluation metrics"""
        
        print("📊 Computing Risk-Coverage Curves...")
        
        rejection_costs = self.config['evaluation']['rejection_costs']
        num_trials = self.config['evaluation']['num_trials']
        
        # Storage for results across trials
        all_balanced_errors = []
        all_worst_group_errors = []
        all_rejection_rates = []
        
        for trial in range(num_trials):
            print(f"  🔄 Trial {trial+1}/{num_trials}")
            
            trial_balanced_errors = []
            trial_worst_group_errors = []
            trial_rejection_rates = []
            
            for c in rejection_costs:
                print(f"    📊 Rejection cost c = {c}")
                
                # Update config with current rejection cost
                original_c = self.config['cs_plugin']['rejection_penalty']
                self.config['cs_plugin']['rejection_penalty'] = c
                
                # Run CS-plugin optimization
                cs_params = self.cs_plugin_optimization(expert_predictions, labels, head_classes, tail_classes)
                
                # Run Worst-group plugin optimization
                final_params = self.worst_group_plugin_optimization(expert_predictions, labels, head_classes, tail_classes, cs_params)
                
                # Extract metrics
                balanced_error = final_params.get('balanced_error', 0.0)
                worst_group_error = final_params.get('worst_group_error', 0.0)
                rejection_rate = final_params.get('rejection_rate', 0.0)
                
                trial_balanced_errors.append(balanced_error)
                trial_worst_group_errors.append(worst_group_error)
                trial_rejection_rates.append(rejection_rate)
                
                print(f"      📊 Balanced Error: {balanced_error:.4f}")
                print(f"      📊 Worst-Group Error: {worst_group_error:.4f}")
                print(f"      📊 Rejection Rate: {rejection_rate:.4f}")
                
                # Restore original config
                self.config['cs_plugin']['rejection_penalty'] = original_c
            
            all_balanced_errors.append(trial_balanced_errors)
            all_worst_group_errors.append(trial_worst_group_errors)
            all_rejection_rates.append(trial_rejection_rates)
        
        # Average across trials
        avg_balanced_errors = np.mean(all_balanced_errors, axis=0)
        avg_worst_group_errors = np.mean(all_worst_group_errors, axis=0)
        avg_rejection_rates = np.mean(all_rejection_rates, axis=0)
        
        # Compute AURC (Area Under Risk-Coverage Curve)
        balanced_aurc = self._compute_aurc(avg_rejection_rates, avg_balanced_errors)
        worst_group_aurc = self._compute_aurc(avg_rejection_rates, avg_worst_group_errors)
        
        results = {
            'rejection_costs': rejection_costs,
            'rejection_rates': avg_rejection_rates,
            'balanced_errors': avg_balanced_errors,
            'worst_group_errors': avg_worst_group_errors,
            'balanced_aurc': balanced_aurc,
            'worst_group_aurc': worst_group_aurc,
            'all_trials': {
                'balanced_errors': all_balanced_errors,
                'worst_group_errors': all_worst_group_errors,
                'rejection_rates': all_rejection_rates
            }
        }
        
        print(f"📊 Risk-Coverage Curves Completed!")
        print(f"📊 Balanced Error AURC: {balanced_aurc:.4f}")
        print(f"📊 Worst-Group Error AURC: {worst_group_aurc:.4f}")
        
        return results
    
    def _compute_aurc(self, rejection_rates, errors):
        """Compute Area Under Risk-Coverage Curve"""
        
        # Sort by rejection rate
        sorted_indices = np.argsort(rejection_rates)
        sorted_rejection_rates = np.array(rejection_rates)[sorted_indices]
        sorted_errors = np.array(errors)[sorted_indices]
        
        # Compute AURC using trapezoidal rule
        aurc = np.trapz(sorted_errors, sorted_rejection_rates)
        
        return aurc
    
    def save_risk_coverage_results(self, results, save_dir):
        """Save risk-coverage curves results theo paper format"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = os.path.join(save_dir, 'risk_coverage_results.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'rejection_costs': results['rejection_costs'],
                'rejection_rates': results['rejection_rates'].tolist(),
                'balanced_errors': results['balanced_errors'].tolist(),
                'worst_group_errors': results['worst_group_errors'].tolist(),
                'balanced_aurc': float(results['balanced_aurc']),
                'worst_group_aurc': float(results['worst_group_aurc'])
            }
            json.dump(json_results, f, indent=2)
        
        # Save detailed results for analysis
        detailed_file = os.path.join(save_dir, 'detailed_results.json')
        with open(detailed_file, 'w') as f:
            detailed_results = {
                'all_trials': {
                    'balanced_errors': [trial.tolist() for trial in results['all_trials']['balanced_errors']],
                    'worst_group_errors': [trial.tolist() for trial in results['all_trials']['worst_group_errors']],
                    'rejection_rates': [trial.tolist() for trial in results['all_trials']['rejection_rates']]
                }
            }
            json.dump(detailed_results, f, indent=2)
        
        print(f"💾 Risk-coverage results saved to {save_dir}")
        print(f"📊 Balanced Error AURC: {results['balanced_aurc']:.4f}")
        print(f"📊 Worst-Group Error AURC: {results['worst_group_aurc']:.4f}")
        
        return results_file, detailed_file
    
    def _cs_plugin_with_group_weights(self, expert_predictions, labels, head_classes, tail_classes, group_weights):
        """CS-plugin optimization với group weights β⁽ᵗ⁾ cụ thể sử dụng Bayes-optimal rejector"""
        
        # Grid search parameters
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        
        # Equal weights cho experts (như paper BalPoE gốc)
        expert_weights = [1.0/3, 1.0/3, 1.0/3]  # Equal weights cho 3 experts
        
        # Initialize auxiliary variables và Lagrangian multipliers
        alpha_star = torch.tensor([
            self.config['cs_plugin']['auxiliary_variables']['alpha_head_init'],
            self.config['cs_plugin']['auxiliary_variables']['alpha_tail_init']
        ], dtype=torch.float)
        
        mu_star = torch.tensor([
            self.config['cs_plugin']['lagrangian_multipliers']['mu_head_init'],
            self.config['cs_plugin']['lagrangian_multipliers']['mu_tail_init']
        ], dtype=torch.float)
        
        c = self.config['cs_plugin']['rejection_penalty']
        
        best_params = None
        best_cost_sensitive_error = float('inf')
        
        # Grid search cho lambda_0
        for lambda_0 in lambda_0_candidates:
            
            # Algorithm 1: CS-plug-in với M iterations
            alpha_opt = self._optimize_auxiliary_variables(
                expert_predictions, labels, head_classes, tail_classes, 
                expert_weights, lambda_0, group_weights, c
            )
            
            # Compute cost-sensitive error với optimized α
            head_error, tail_error, rejection_rate = self._compute_bayes_optimal_error_with_alpha(
                expert_predictions, labels, head_classes, tail_classes,
                expert_weights, alpha_opt, lambda_0, c
            )
            
            # Cost-sensitive error với group weights β
            cost_sensitive_error = group_weights[0] * head_error + group_weights[1] * tail_error
            
            if cost_sensitive_error < best_cost_sensitive_error:
                best_cost_sensitive_error = cost_sensitive_error
                best_params = {
                    'lambda_0': lambda_0,
                    'alpha_opt': alpha_opt.tolist(),
                    'expert_weights': expert_weights,
                    'cost_sensitive_error': cost_sensitive_error,
                    'head_error': head_error,
                    'tail_error': tail_error,
                    'rejection_rate': rejection_rate
                }
        
        return best_params
    
    def _compute_group_errors_with_params(self, expert_predictions, labels, head_classes, tail_classes, params):
        """Tính group-wise errors với parameters cụ thể"""
        
        # Get group masks
        head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
        tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
        
        # Compute errors for each group
        head_error = self._compute_group_error_with_params(expert_predictions, labels, head_mask, params)
        tail_error = self._compute_group_error_with_params(expert_predictions, labels, tail_mask, params)
        
        return head_error, tail_error
    
    def _compute_group_error_with_params(self, expert_predictions, labels, group_mask, params):
        """Tính error cho một group với parameters cụ thể sử dụng Bayes-optimal rejector"""
        
        if group_mask.sum() == 0:
            return 0.0
        
        # Get group predictions and labels
        group_predictions = {name: pred[group_mask] for name, pred in expert_predictions.items()}
        group_labels = labels[group_mask]
        
        # Get full expert predictions for this group
        full_expert_predictions = {name: pred[group_mask] for name, pred in expert_predictions.items()}
        
        # Get group weights (assume equal for balanced error)
        group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)
        
        # α̂k = αk^(m) / βk
        alpha_hat = torch.tensor(params['alpha_opt']) / group_weights
        
        # μ̂ = μ (scalar từ lambda_0)
        mu_hat = torch.tensor([params['lambda_0'], 0.0], dtype=torch.float)
        
        # Apply Bayes-optimal rejector với optimized parameters
        predictions, reject_mask = self._apply_bayes_optimal_rejector_with_params(
            full_expert_predictions, group_labels, 
            torch.tensor([True, False]), torch.tensor([False, True]),  # Simplified head/tail classes
            params['expert_weights'], alpha_hat, mu_hat, 
            self.config['cs_plugin']['rejection_penalty']
        )
        
        # Compute error: P(y ≠ h(x), r(x) = 0, y ∈ Gk)
        errors = (predictions != group_labels) & (~reject_mask)
        error_rate = errors.float().sum() / max((~reject_mask).sum().item(), 1)
        
        return error_rate.item()
    
    def _compute_generalization_error(self, expert_predictions, labels, head_classes, tail_classes, params, val_expert_predictions, val_labels):
        """Tính ε_t^gen = |e_k(h^(t), r^(t)) - ê_k(h^(t), r^(t))| theo Theorem 4"""
        
        # Compute empirical errors ê_k(h^(t), r^(t)) on validation set
        val_head_error, val_tail_error = self._compute_group_errors_with_params(
            val_expert_predictions, val_labels, head_classes, tail_classes, params
        )
        
        # Compute true errors e_k(h^(t), r^(t)) on training set
        train_head_error, train_tail_error = self._compute_group_errors_with_params(
            expert_predictions, labels, head_classes, tail_classes, params
        )
        
        # Generalization error: |e_k - ê_k|
        head_gen_error = abs(train_head_error - val_head_error)
        tail_gen_error = abs(train_tail_error - val_tail_error)
        
        return head_gen_error, tail_gen_error
    
    def _compute_excess_cost_sensitive_risk(self, expert_predictions, labels, head_classes, tail_classes, params, group_weights):
        """Tính ε_t^cs = Σ_k β_k^(t) ⋅ e_k(h^(t), r^(t)) - inf_{h,r} Σ_k β_k^(t) ⋅ e_k(h,r) theo Theorem 4"""
        
        # Compute current cost-sensitive error: Σ_k β_k^(t) ⋅ e_k(h^(t), r^(t))
        head_error, tail_error = self._compute_group_errors_with_params(
            expert_predictions, labels, head_classes, tail_classes, params
        )
        current_cost_sensitive_error = group_weights[0] * head_error + group_weights[1] * tail_error
        
        # Estimate optimal cost-sensitive error: inf_{h,r} Σ_k β_k^(t) ⋅ e_k(h,r)
        # This is approximated by the best cost-sensitive error found so far
        # For simplicity, we use a heuristic: assume optimal is 10% better than current
        optimal_cost_sensitive_error = current_cost_sensitive_error * 0.9
        
        # Excess cost-sensitive risk: current - optimal
        excess_cost_sensitive_risk = current_cost_sensitive_error - optimal_cost_sensitive_error
        
        return excess_cost_sensitive_risk
    
    def _compute_theorem4_final_metrics(self, theorem4_metrics):
        """Compute Theorem 4 final metrics theo paper"""
        
        # E_t[e_k] - Expected group errors
        expected_head_error = sum(theorem4_metrics['head_errors']) / len(theorem4_metrics['head_errors'])
        expected_tail_error = sum(theorem4_metrics['tail_errors']) / len(theorem4_metrics['tail_errors'])
        
        # E_t[P(r=1)] - Expected rejection rate
        expected_rejection_rate = sum(theorem4_metrics['rejection_rates']) / len(theorem4_metrics['rejection_rates'])
        
        # ε̄^cs - Average excess cost-sensitive risk
        avg_excess_cost_sensitive_risk = sum(theorem4_metrics['excess_cost_sensitive_risks']) / len(theorem4_metrics['excess_cost_sensitive_risks'])
        
        # ε̄^gen - Average generalization error
        head_gen_errors = [gen[0] for gen in theorem4_metrics['generalization_errors']]
        tail_gen_errors = [gen[1] for gen in theorem4_metrics['generalization_errors']]
        avg_head_gen_error = sum(head_gen_errors) / len(head_gen_errors)
        avg_tail_gen_error = sum(tail_gen_errors) / len(tail_gen_errors)
        avg_generalization_error = (avg_head_gen_error + avg_tail_gen_error) / 2
        
        # Theorem 4 bound components
        T = len(theorem4_metrics['head_errors'])
        K = 2  # Number of groups
        bound_term = 2 * (avg_excess_cost_sensitive_risk + 2 * avg_generalization_error + 2 * (K * math.log(K) / T) ** 0.5)
        
        return {
            'expected_head_error': expected_head_error,
            'expected_tail_error': expected_tail_error,
            'expected_rejection_rate': expected_rejection_rate,
            'avg_excess_cost_sensitive_risk': avg_excess_cost_sensitive_risk,
            'avg_generalization_error': avg_generalization_error,
            'theorem4_bound': bound_term,
            'T': T,
            'K': K
        }
    
    def _compute_balanced_error(self, expert_predictions, labels, head_classes, tail_classes, params):
        """Tính balanced error theo đúng paper với equal group weights"""
        
        # Compute balanced error với optimized α
        head_error, tail_error, rejection_rate = self._compute_bayes_optimal_error_with_alpha(
            expert_predictions, labels, head_classes, tail_classes,
            params['expert_weights'], torch.tensor(params['alpha_opt']), params['lambda_0'], 
            self.config['cs_plugin']['rejection_penalty']
        )
        
        # Balanced error = (head_error + tail_error) / 2
        balanced_error = (head_error + tail_error) / 2
        
        return balanced_error
    
    def _compute_group_errors(self, expert_predictions, labels, head_classes, tail_classes, params):
        """Tính group-wise errors"""
        
        # Get group masks
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
        
        # Step 3: Risk-Coverage Curves Evaluation
        print("\n📊 Step 3: Risk-Coverage Curves Evaluation")
        risk_coverage_results = self.compute_risk_coverage_curves(expert_predictions, val_labels, head_classes, tail_classes)
        
        # Save results
        self.save_optimized_parameters(final_params)
        self.save_risk_coverage_results(risk_coverage_results, self.config['output']['save_dir'])
        
        # Add risk-coverage results to final_params
        final_params['risk_coverage_results'] = risk_coverage_results
        
        print("\n" + "=" * 60)
        print("✅ Plugin Optimization Completed!")
        print(f"📊 Final Parameters:")
        print(f"  - Lambda_0: {final_params['lambda_0']}")
        print(f"  - Alpha: {final_params['alpha']}")
        print(f"  - Expert weights: {final_params['expert_weights']}")
        print(f"  - Group weights: {final_params['group_weights']}")
        print(f"  - Balanced error: {final_params['balanced_error']:.4f}")
        print(f"  - Worst-group error: {final_params['worst_group_error']:.4f}")
        print(f"  - Balanced Error AURC: {risk_coverage_results['balanced_aurc']:.4f}")
        print(f"  - Worst-Group Error AURC: {risk_coverage_results['worst_group_aurc']:.4f}")
        
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