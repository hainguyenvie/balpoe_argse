"""
Plugin Optimization Methods
Implement CS-plugin và Worst-group Plugin algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class PluginParameters:
    """Lưu trữ các tham số plugin đã tối ưu"""
    lambda_0: float
    alpha: float
    expert_weights: List[float]  # [w_head, w_balanced, w_tail]
    group_weights: List[float]   # [w_head_group, w_tail_group]
    rejection_threshold: float
    balanced_error: float
    worst_group_error: float


class ExpertEnsemble:
    """Quản lý ensemble của các experts"""
    
    def __init__(self, expert_checkpoints: Dict[str, str], device='cuda'):
        self.device = device
        self.experts = {}
        self.tau_values = [0, 1.0, 2.0]
        self.expert_names = ['head_expert', 'balanced_expert', 'tail_expert']
        
        # Load expert models
        for name, checkpoint_path in expert_checkpoints.items():
            print(f"📂 Loading {name} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract expert from BalPoE checkpoint
            expert_model = self._extract_expert_from_balpoe(checkpoint, name)
            expert_model.eval()
            self.experts[name] = expert_model
            
        print(f"✅ Loaded {len(self.experts)} experts")
    
    def _extract_expert_from_balpoe(self, checkpoint: Dict, expert_name: str):
        """Trích xuất expert riêng lẻ từ BalPoE checkpoint"""
        # Implementation depends on how experts are stored in BalPoE
        # This is a simplified version - cần adjust based on actual BalPoE structure
        expert_idx = self.expert_names.index(expert_name)
        
        # Create single expert model (simplified)
        from model.model import ResNet32Model
        expert_model = ResNet32Model(
            num_classes=100,
            num_experts=1,
            use_norm=True
        )
        
        # Load expert weights (cần implement logic để extract từ BalPoE)
        # expert_model.load_state_dict(expert_state_dict)
        
        return expert_model
    
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Dự đoán xác suất từ tất cả experts"""
        with torch.no_grad():
            expert_probs = []
            
            for expert_name, expert_model in self.experts.items():
                # Get expert prediction
                expert_output = expert_model(x)
                expert_probs.append(F.softmax(expert_output, dim=1))
            
            # Stack predictions: [batch_size, num_experts, num_classes]
            expert_probs = torch.stack(expert_probs, dim=1)
            
        return expert_probs


class CSPluginOptimizer:
    """Cost-Sensitive Plugin Optimizer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lambda_0_candidates = config['cs_plugin']['lambda_0_candidates']
        self.alpha_iterations = config['cs_plugin']['alpha_search_iterations']
        self.weight_search_space = config['cs_plugin']['weight_search_space']
        
    def optimize(self, expert_ensemble: ExpertEnsemble, 
                val_data_loader, val_labels: torch.Tensor) -> PluginParameters:
        """Tối ưu CS-plugin parameters"""
        
        print("🔍 Bắt đầu CS-plugin optimization...")
        
        # Get expert predictions on validation set
        expert_predictions = self._get_expert_predictions(expert_ensemble, val_data_loader)
        
        best_params = None
        best_balanced_error = float('inf')
        
        # Grid search over lambda_0
        for lambda_0 in self.lambda_0_candidates:
            print(f"  🔍 Testing lambda_0 = {lambda_0}")
            
            # Grid search over expert weights
            for w_head in self.weight_search_space['w_head']:
                for w_balanced in self.weight_search_space['w_balanced']:
                    for w_tail in self.weight_search_space['w_tail']:
                        
                        # Normalize weights
                        total_weight = w_head + w_balanced + w_tail
                        if total_weight == 0:
                            continue
                            
                        expert_weights = [w_head/total_weight, w_balanced/total_weight, w_tail/total_weight]
                        
                        # Find optimal alpha using power iteration
                        alpha = self._find_optimal_alpha(
                            expert_predictions, val_labels, 
                            lambda_0, expert_weights
                        )
                        
                        # Evaluate balanced error
                        balanced_error = self._evaluate_balanced_error(
                            expert_predictions, val_labels,
                            lambda_0, alpha, expert_weights
                        )
                        
                        if balanced_error < best_balanced_error:
                            best_balanced_error = balanced_error
                            best_params = PluginParameters(
                                lambda_0=lambda_0,
                                alpha=alpha,
                                expert_weights=expert_weights,
                                group_weights=[0.5, 0.5],  # Will be updated in worst-group optimization
                                rejection_threshold=0.5,  # Default threshold
                                balanced_error=balanced_error,
                                worst_group_error=0.0  # Will be computed later
                            )
                            
                            print(f"    ✅ New best: balanced_error = {balanced_error:.4f}")
        
        print(f"✅ CS-plugin optimization completed!")
        print(f"📊 Best balanced error: {best_balanced_error:.4f}")
        
        return best_params
    
    def _get_expert_predictions(self, expert_ensemble: ExpertEnsemble, 
                              data_loader) -> torch.Tensor:
        """Lấy predictions từ tất cả experts"""
        expert_predictions = []
        
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(expert_ensemble.device)
            batch_predictions = expert_ensemble.predict_probs(batch_data)
            expert_predictions.append(batch_predictions.cpu())
        
        return torch.cat(expert_predictions, dim=0)
    
    def _find_optimal_alpha(self, expert_predictions: torch.Tensor, 
                          labels: torch.Tensor, lambda_0: float, 
                          expert_weights: List[float]) -> float:
        """Tìm alpha tối ưu bằng power iteration"""
        
        # Simplified power iteration (cần implement đầy đủ theo paper)
        alpha = 0.5  # Initial guess
        
        for iteration in range(self.alpha_iterations):
            # Compute cost-sensitive error với alpha hiện tại
            error = self._compute_cost_sensitive_error(
                expert_predictions, labels, lambda_0, alpha, expert_weights
            )
            
            # Update alpha (simplified update rule)
            alpha = alpha * 0.9 + 0.1 * (1 - error)
            alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        
        return alpha
    
    def _compute_cost_sensitive_error(self, expert_predictions: torch.Tensor,
                                    labels: torch.Tensor, lambda_0: float,
                                    alpha: float, expert_weights: List[float]) -> float:
        """Tính cost-sensitive error"""
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions[:, 0, :])
        for i, weight in enumerate(expert_predictions):
            ensemble_pred += weight * expert_predictions[:, i, :]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < alpha
        
        # Compute error (simplified)
        correct_mask = (predictions == labels) & (~reject_mask)
        error = 1.0 - correct_mask.float().mean()
        
        return error.item()
    
    def _evaluate_balanced_error(self, expert_predictions: torch.Tensor,
                               labels: torch.Tensor, lambda_0: float,
                               alpha: float, expert_weights: List[float]) -> float:
        """Đánh giá balanced error"""
        return self._compute_cost_sensitive_error(
            expert_predictions, labels, lambda_0, alpha, expert_weights
        )


class WorstGroupPluginOptimizer:
    """Worst-Group Plugin Optimizer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_iterations = config['worst_group_plugin']['max_iterations']
        self.step_size = config['worst_group_plugin']['step_size']
        self.initial_group_weights = config['worst_group_plugin']['initial_group_weights']
        
    def optimize(self, expert_ensemble: ExpertEnsemble, 
                val_data_loader, val_labels: torch.Tensor,
                cs_plugin_params: PluginParameters) -> PluginParameters:
        """Tối ưu worst-group plugin"""
        
        print("🔍 Bắt đầu Worst-group plugin optimization...")
        
        # Get expert predictions
        expert_predictions = self._get_expert_predictions(expert_ensemble, val_data_loader)
        
        # Initialize group weights
        group_weights = torch.tensor(self.initial_group_weights, dtype=torch.float)
        
        best_params = cs_plugin_params
        best_worst_group_error = float('inf')
        
        for iteration in range(self.max_iterations):
            print(f"  🔄 Iteration {iteration+1}/{self.max_iterations}")
            
            # Compute group-wise errors
            head_error, tail_error = self._compute_group_errors(
                expert_predictions, val_labels, cs_plugin_params
            )
            
            # Update group weights using exponentiated gradient
            group_weights = self._update_group_weights(
                group_weights, head_error, tail_error
            )
            
            # Evaluate worst-group error
            worst_group_error = max(head_error, tail_error)
            
            if worst_group_error < best_worst_group_error:
                best_worst_group_error = worst_group_error
                best_params = PluginParameters(
                    lambda_0=cs_plugin_params.lambda_0,
                    alpha=cs_plugin_params.alpha,
                    expert_weights=cs_plugin_params.expert_weights,
                    group_weights=group_weights.tolist(),
                    rejection_threshold=cs_plugin_params.rejection_threshold,
                    balanced_error=cs_plugin_params.balanced_error,
                    worst_group_error=worst_group_error
                )
                
                print(f"    ✅ New best: worst_group_error = {worst_group_error:.4f}")
        
        print(f"✅ Worst-group plugin optimization completed!")
        print(f"📊 Best worst-group error: {best_worst_group_error:.4f}")
        
        return best_params
    
    def _get_expert_predictions(self, expert_ensemble: ExpertEnsemble, 
                              data_loader) -> torch.Tensor:
        """Lấy predictions từ experts"""
        expert_predictions = []
        
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(expert_ensemble.device)
            batch_predictions = expert_ensemble.predict_probs(batch_data)
            expert_predictions.append(batch_predictions.cpu())
        
        return torch.cat(expert_predictions, dim=0)
    
    def _compute_group_errors(self, expert_predictions: torch.Tensor,
                               labels: torch.Tensor, params: PluginParameters) -> Tuple[float, float]:
        """Tính group-wise errors"""
        
        # Simplified group error computation
        # Cần implement logic để phân chia head/tail groups
        head_mask = labels < 50  # Simplified: first 50 classes as head
        tail_mask = labels >= 50  # Simplified: last 50 classes as tail
        
        # Compute errors for each group
        head_error = self._compute_group_error(expert_predictions, labels, head_mask, params)
        tail_error = self._compute_group_error(expert_predictions, labels, tail_mask, params)
        
        return head_error, tail_error
    
    def _compute_group_error(self, expert_predictions: torch.Tensor,
                           labels: torch.Tensor, group_mask: torch.Tensor,
                           params: PluginParameters) -> float:
        """Tính error cho một group"""
        
        if group_mask.sum() == 0:
            return 0.0
        
        # Get group predictions and labels
        group_predictions = expert_predictions[group_mask]
        group_labels = labels[group_mask]
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(group_predictions[:, 0, :])
        for i, weight in enumerate(params.expert_weights):
            ensemble_pred += weight * group_predictions[:, i, :]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < params.alpha
        
        # Compute error
        correct_mask = (predictions == group_labels) & (~reject_mask)
        error = 1.0 - correct_mask.float().mean()
        
        return error.item()
    
    def _update_group_weights(self, group_weights: torch.Tensor,
                            head_error: float, tail_error: float) -> torch.Tensor:
        """Cập nhật group weights bằng exponentiated gradient"""
        
        # Compute gradients
        grad_head = head_error
        grad_tail = tail_error
        
        # Update weights
        log_weights = torch.log(group_weights + 1e-8)
        log_weights[0] -= self.step_size * grad_head
        log_weights[1] -= self.step_size * grad_tail
        
        # Normalize
        group_weights = torch.softmax(log_weights, dim=0)
        
        return group_weights


class PluginOptimizer:
    """Main Plugin Optimizer - kết hợp CS-plugin và Worst-group plugin"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.cs_optimizer = CSPluginOptimizer(self.config)
        self.worst_group_optimizer = WorstGroupPluginOptimizer(self.config)
        
    def optimize(self, expert_checkpoints: Dict[str, str], 
                val_data_loader, val_labels: torch.Tensor) -> PluginParameters:
        """Tối ưu toàn bộ plugin pipeline"""
        
        print("🚀 Bắt đầu Plugin Optimization Pipeline")
        print("=" * 60)
        
        # Load expert ensemble
        expert_ensemble = ExpertEnsemble(expert_checkpoints)
        
        # Step 1: CS-plugin optimization
        print("\n📊 Step 1: CS-plugin Optimization")
        cs_params = self.cs_optimizer.optimize(expert_ensemble, val_data_loader, val_labels)
        
        # Step 2: Worst-group plugin optimization
        print("\n📊 Step 2: Worst-group Plugin Optimization")
        final_params = self.worst_group_optimizer.optimize(
            expert_ensemble, val_data_loader, val_labels, cs_params
        )
        
        print("\n" + "=" * 60)
        print("✅ Plugin Optimization Completed!")
        print(f"📊 Final Parameters:")
        print(f"  - Lambda_0: {final_params.lambda_0}")
        print(f"  - Alpha: {final_params.alpha}")
        print(f"  - Expert weights: {final_params.expert_weights}")
        print(f"  - Group weights: {final_params.group_weights}")
        print(f"  - Balanced error: {final_params.balanced_error:.4f}")
        print(f"  - Worst-group error: {final_params.worst_group_error:.4f}")
        
        return final_params
