#!/usr/bin/env python3
"""
Giai đoạn 2: Tối ưu hóa Plugin Parameters với Gating Network
Mục tiêu: Sử dụng Gating Network để học expert weights thay vì grid search
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

# Import Gating Network
from gating_network import create_gating_network, create_gating_trainer


class MoEPluginGatingOptimizer:
    """
    MoE-Plugin Optimizer với Gating Network
    Thay thế grid search bằng neural network để học expert weights
    """
    
    def __init__(self, experts_dir: str, config_path: str, seed: int = 1):
        self.experts_dir = Path(experts_dir)
        self.config_path = config_path
        self.seed = seed
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        seed_everything(seed)
        
        print(f"🚀 Khởi tạo MoE-Plugin Gating Optimizer")
        print(f"📂 Experts directory: {self.experts_dir}")
        print(f"📊 Dataset: {self.config['dataset']['name']}")
        print(f"🎯 CS-plugin candidates: {self.config['cs_plugin']['lambda_0_candidates']}")
        print(f"🔄 Worst-group iterations: {self.config['worst_group_plugin']['max_iterations']}")
        print(f"🧠 Gating Network: {self.config['gating_network']['hidden_dims']}")
        
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
    
    def _detect_feature_dimension(self):
        """Detect feature dimension từ BalPoE model"""
        print("🔍 Detecting feature dimension...")
        
        # Get a sample batch
        sample_batch = next(iter(self.val_data_loader))
        sample_data, _ = sample_batch
        sample_data = sample_data[:1]  # Just one sample
        
        # Get features from balanced expert
        with torch.no_grad():
            balanced_output = self.expert_models['balanced_expert'](sample_data)
            
            if isinstance(balanced_output, dict) and 'feat' in balanced_output:
                features = balanced_output['feat']
            else:
                # Fallback: try to get features from model
                try:
                    features = self.expert_models['balanced_expert'].get_features(sample_data)
                except:
                    # If no get_features method, use the last layer before classifier
                    # This is a simplified approach - might need adjustment based on actual model
                    features = balanced_output['output'] if isinstance(balanced_output, dict) else balanced_output
        
        # Get feature dimension
        if len(features.shape) > 2:
            # If features are 2D+ (e.g., [batch, seq, dim]), flatten
            features = features.view(features.size(0), -1)
        
        feature_dim = features.shape[1]
        print(f"📊 Feature shape: {features.shape}")
        print(f"📊 Feature dimension: {feature_dim}")
        
        return feature_dim
    
    def train_gating_network(self):
        """Train Gating Network để học expert weights"""
        print("🧠 Training Gating Network...")
        
        # First, detect feature dimension from a sample
        feature_dim = self._detect_feature_dimension()
        print(f"🔍 Detected feature dimension: {feature_dim}")
        
        # Create gating network with detected dimension
        gating_config = self.config['gating_network'].copy()
        gating_config['input_dim'] = feature_dim  # Override with detected dimension
        
        self.gating_network = create_gating_network(
            input_dim=feature_dim,
            config=gating_config
        )
        
        # Create trainer
        trainer_config = self.config['gating_trainer']
        self.gating_trainer = create_gating_trainer(self.gating_network, trainer_config)
        
        # Train gating network
        training_results = self.gating_trainer.train(
            data_loader=self.val_data_loader,
            expert_models=self.expert_models,
            labels=self._get_all_labels(self.val_data_loader),
            device='cpu'
        )
        
        print(f"✅ Gating Network training completed!")
        print(f"📊 Best loss: {training_results['best_loss']:.4f}")
        print(f"📊 Final loss: {training_results['final_loss']:.4f}")
        
        return training_results
    
    def _get_all_labels(self, data_loader):
        """Get all labels from data loader"""
        all_labels = []
        for _, batch_labels in data_loader:
            all_labels.append(batch_labels)
        return torch.cat(all_labels, dim=0)
    
    def get_gated_expert_predictions(self, data_loader):
        """Lấy predictions từ experts với gating weights"""
        print("🔍 Getting gated expert predictions...")
        
        gated_predictions = []
        all_labels = []
        
        self.gating_network.eval()
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to('cpu')
                all_labels.append(batch_labels)
                
                # Get expert predictions (frozen)
                expert_predictions = {}
                for name, model in self.expert_models.items():
                    expert_output = model(batch_data)
                    
                    # Handle BalPoE model output
                    if isinstance(expert_output, dict):
                        if 'logits' in expert_output:
                            all_expert_logits = expert_output['logits']
                            expert_idx = {'head_expert': 0, 'balanced_expert': 1, 'tail_expert': 2}[name]
                            expert_logits = all_expert_logits[:, expert_idx, :]
                        elif 'output' in expert_output:
                            expert_logits = expert_output['output']
                        else:
                            expert_logits = list(expert_output.values())[0]
                    else:
                        expert_logits = expert_output
                    
                    expert_probs = torch.softmax(expert_logits, dim=1)
                    expert_predictions[name] = expert_probs
                
                # Get feature vectors (use balanced expert)
                balanced_output = self.expert_models['balanced_expert'](batch_data)
                if isinstance(balanced_output, dict) and 'feat' in balanced_output:
                    features = balanced_output['feat']
                else:
                    # Fallback: try to get features from model
                    try:
                        features = self.expert_models['balanced_expert'].get_features(batch_data)
                    except:
                        # If no get_features method, use the last layer before classifier
                        features = balanced_output['output'] if isinstance(balanced_output, dict) else balanced_output
                
                # Ensure features are 2D
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                
                # Get gating weights
                expert_weights = self.gating_network(features)  # [batch_size, 3]
                
                # Compute mixed probabilities
                mixed_probs = torch.zeros_like(expert_predictions['head_expert'])
                for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], 
                                                     expert_weights.T)):
                    mixed_probs += weight.unsqueeze(1) * expert_predictions[name]
                
                gated_predictions.append(mixed_probs.cpu())
        
        # Combine predictions
        gated_predictions = torch.cat(gated_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"✅ Gated predictions shape: {gated_predictions.shape}")
        print(f"✅ Labels shape: {all_labels.shape}")
        
        return gated_predictions, all_labels
    
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
    
    def cs_plugin_optimization(self, gated_predictions, labels, head_classes, tail_classes):
        """
        Thuật toán 1: CS-plugin với Gated predictions
        Không cần grid search cho weights nữa, chỉ cần tìm λ₀ và α
        """
        print("🔍 CS-plugin Optimization với Gated predictions...")
        
        # Grid search chỉ cho lambda_0
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        
        best_params = None
        best_balanced_error = float('inf')
        
        for lambda_0 in lambda_0_candidates:
            print(f"  🔍 Testing λ₀ = {lambda_0}")
            
            # Tìm alpha tối ưu bằng power iteration
            alpha = self._find_optimal_alpha(
                gated_predictions, labels, lambda_0, head_classes, tail_classes
            )
            
            # Đánh giá balanced error
            balanced_error = self._evaluate_balanced_error(
                gated_predictions, labels, lambda_0, alpha, head_classes, tail_classes
            )
            
            print(f"    📊 α = {alpha:.4f}, Balanced Error = {balanced_error:.4f}")
            
            if balanced_error < best_balanced_error:
                best_balanced_error = balanced_error
                best_params = {
                    'lambda_0': lambda_0,
                    'alpha': alpha,
                    'balanced_error': balanced_error
                }
                
                print(f"    ✅ New best: balanced_error = {balanced_error:.4f}")
        
        print(f"✅ CS-plugin optimization completed!")
        print(f"📊 Best balanced error: {best_balanced_error:.4f}")
        
        return best_params
    
    def _find_optimal_alpha(self, gated_predictions, labels, lambda_0, head_classes, tail_classes):
        """Tìm alpha tối ưu bằng power iteration (M=10 iterations)"""
        
        # Initialize alpha
        alpha = 0.5  # Initial guess
        M = self.config['cs_plugin']['alpha_search_iterations']
        
        # Default group weights for balanced error (equal weights)
        group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)
        
        for iteration in range(M):
            # Compute cost-sensitive error với alpha hiện tại
            error = self._compute_cost_sensitive_error(
                gated_predictions, labels, lambda_0, alpha, head_classes, tail_classes, group_weights
            )
            
            # Power iteration update rule
            target_error = 0.5
            learning_rate = 0.1
            alpha_update = learning_rate * (target_error - error)
            alpha = alpha + alpha_update
            
            # Clamp alpha to reasonable range [0.1, 0.9]
            alpha = max(0.1, min(0.9, alpha))
        
        return alpha
    
    def _compute_cost_sensitive_error(self, gated_predictions, labels, lambda_0, alpha, head_classes, tail_classes, group_weights):
        """Tính cost-sensitive error với gated predictions"""
        
        # Apply rejection rule
        max_probs, predictions = torch.max(gated_predictions, dim=1)
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
    
    def _evaluate_balanced_error(self, gated_predictions, labels, lambda_0, alpha, head_classes, tail_classes):
        """Đánh giá balanced error"""
        return self._compute_cost_sensitive_error(
            gated_predictions, labels, lambda_0, alpha, head_classes, tail_classes, 
            torch.tensor([0.5, 0.5])
        )
    
    def worst_group_plugin_optimization(self, gated_predictions, labels, head_classes, tail_classes, cs_params):
        """
        Thuật toán 2: Worst-group Plugin với Gated predictions
        """
        print("🔍 Worst-group Plugin Optimization với Gated predictions...")
        
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
            
            # Tìm (h⁽ᵗ⁾, r⁽ᵗ⁾) tối ưu cho cost-sensitive error với β⁽ᵗ⁾
            current_params = self._cs_plugin_with_group_weights(
                gated_predictions, labels, head_classes, tail_classes, group_weights
            )
            
            # Compute group-wise errors với classifier (h⁽ᵗ⁾, r⁽ᵗ⁾)
            head_error, tail_error = self._compute_group_errors_with_params(
                gated_predictions, labels, head_classes, tail_classes, current_params
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
    
    def _cs_plugin_with_group_weights(self, gated_predictions, labels, head_classes, tail_classes, group_weights):
        """CS-plugin optimization với group weights β⁽ᵗ⁾ cụ thể"""
        
        # Grid search chỉ cho lambda_0
        lambda_0_candidates = self.config['cs_plugin']['lambda_0_candidates']
        
        best_params = None
        best_cost_sensitive_error = float('inf')
        
        for lambda_0 in lambda_0_candidates:
            # Tìm alpha tối ưu với group weights β⁽ᵗ⁾
            alpha = self._find_optimal_alpha(
                gated_predictions, labels, lambda_0, head_classes, tail_classes
            )
            
            # Đánh giá cost-sensitive error với β⁽ᵗ⁾
            cost_sensitive_error = self._compute_cost_sensitive_error(
                gated_predictions, labels, lambda_0, alpha, head_classes, tail_classes, group_weights
            )
            
            if cost_sensitive_error < best_cost_sensitive_error:
                best_cost_sensitive_error = cost_sensitive_error
                best_params = {
                    'lambda_0': lambda_0,
                    'alpha': alpha,
                    'cost_sensitive_error': cost_sensitive_error
                }
        
        return best_params
    
    def _compute_group_errors_with_params(self, gated_predictions, labels, head_classes, tail_classes, params):
        """Tính group-wise errors với parameters cụ thể"""
        
        # Get group masks
        head_mask = torch.tensor([head_classes[label.item()] for label in labels])
        tail_mask = torch.tensor([tail_classes[label.item()] for label in labels])
        
        # Compute errors for each group
        head_error = self._compute_group_error_with_params(gated_predictions, labels, head_mask, params)
        tail_error = self._compute_group_error_with_params(gated_predictions, labels, tail_mask, params)
        
        return head_error, tail_error
    
    def _compute_group_error_with_params(self, gated_predictions, labels, group_mask, params):
        """Tính error cho một group với parameters cụ thể"""
        
        if group_mask.sum() == 0:
            return 0.0
        
        # Get group predictions and labels
        group_predictions = gated_predictions[group_mask]
        group_labels = labels[group_mask]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(group_predictions, dim=1)
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
        """Chạy toàn bộ optimization pipeline với Gating Network"""
        print("🚀 Bắt đầu Plugin Optimization Pipeline với Gating Network")
        print("=" * 60)
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Load expert models
        self.load_expert_models()
        
        # Train gating network
        gating_results = self.train_gating_network()
        
        # Get gated expert predictions
        gated_predictions, val_labels = self.get_gated_expert_predictions(self.val_data_loader)
        
        # Define groups
        head_classes, tail_classes = self.define_groups(val_labels)
        
        # Step 1: CS-plugin optimization
        print("\n📊 Step 1: CS-plugin Optimization với Gated predictions")
        cs_params = self.cs_plugin_optimization(gated_predictions, val_labels, head_classes, tail_classes)
        
        # Step 2: Worst-group plugin optimization
        print("\n📊 Step 2: Worst-group Plugin Optimization với Gated predictions")
        final_params = self.worst_group_plugin_optimization(gated_predictions, val_labels, head_classes, tail_classes, cs_params)
        
        print("\n" + "=" * 60)
        print("✅ Plugin Optimization với Gating Network Completed!")
        print(f"📊 Final Parameters:")
        print(f"  - Lambda_0: {final_params['lambda_0']}")
        print(f"  - Alpha: {final_params['alpha']}")
        print(f"  - Group weights: {final_params['group_weights']}")
        print(f"  - Balanced error: {final_params['balanced_error']:.4f}")
        print(f"  - Worst-group error: {final_params['worst_group_error']:.4f}")
        print(f"📊 Gating Network Results:")
        print(f"  - Best training loss: {gating_results['best_loss']:.4f}")
        print(f"  - Final training loss: {gating_results['final_loss']:.4f}")
        
        return final_params, gating_results
    
    def save_optimized_parameters(self, params, gating_results):
        """Lưu optimized parameters và gating network"""
        print("💾 Lưu optimized parameters và gating network...")
        
        save_dir = Path(self.config['output']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save gating network
        gating_save_dir = save_dir / 'gating_network'
        self.gating_trainer.save_model(gating_save_dir)
        
        # Convert to dict for JSON serialization
        params_dict = {
            'lambda_0': params['lambda_0'],
            'alpha': params['alpha'],
            'group_weights': params['group_weights'],
            'rejection_threshold': params['alpha'],
            'balanced_error': params['balanced_error'],
            'worst_group_error': params['worst_group_error'],
            'gating_network': {
                'save_dir': str(gating_save_dir),
                'best_loss': gating_results['best_loss'],
                'final_loss': gating_results['final_loss']
            },
            'config': self.config
        }
        
        # Save parameters
        params_file = save_dir / 'optimized_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"✅ Parameters saved to: {params_file}")
        print(f"✅ Gating network saved to: {gating_save_dir}")
        
        return save_dir


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Optimize Plugin Parameters với Gating Network')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to plugin optimization config file')
    parser.add_argument('--experts_dir', type=str, required=True,
                       help='Directory containing expert checkpoints')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = MoEPluginGatingOptimizer(
        experts_dir=args.experts_dir,
        config_path=args.config,
        seed=args.seed
    )
    
    # Run optimization
    optimized_params, gating_results = optimizer.optimize()
    
    # Save optimized parameters
    save_dir = optimizer.save_optimized_parameters(optimized_params, gating_results)
    
    print(f"\n🎉 Giai đoạn 2 với Gating Network hoàn thành!")
    print(f"📁 Optimized parameters: {save_dir}")
    print(f"🧠 Gating network: {save_dir}/gating_network")
    print(f"🔧 Tiếp theo: python stage3_evaluate.py --plugin_checkpoint {save_dir}/optimized_parameters.json")


if __name__ == '__main__':
    main()
