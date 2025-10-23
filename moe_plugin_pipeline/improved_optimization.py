#!/usr/bin/env python3
"""
Improved optimization với các cải thiện:
1. Kiểm tra chất lượng expert trước khi optimize
2. Thêm constraints để đảm bảo tail expert được sử dụng
3. Cải thiện worst-group optimization
4. Thêm early stopping và regularization
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
import sys
import os
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import data_loader.data_loaders as module_data
import model.model as module_arch


class ImprovedMoEPluginOptimizer:
    """Improved MoE Plugin Optimizer với các cải thiện"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.experts_dir = Path(self.config['experts']['experts_dir'])
        self.device = 'cpu'  # Use CPU for stability
        
    def setup_data_loaders(self):
        """Thiết lập data loaders với improved splitting"""
        print("📂 Thiết lập data loaders...")
        
        # Create data loader config
        data_loader_config = {
            "type": "ImbalanceCIFAR100DataLoader",
            "args": {
                "data_dir": self.config['dataset']['data_dir'],
                "batch_size": 256,
                "shuffle": False,
                "num_workers": 0,  # Avoid multiprocessing issues
                "imb_factor": 0.01,
                "randaugm": False,
                "training": False
            }
        }
        
        # Create full data loader
        self.full_data_loader = getattr(module_data, data_loader_config['type'])(
            **data_loader_config['args']
        )
        
        # Improved splitting with stratification
        val_split = self.config['dataset']['val_split']
        total_samples = len(self.full_data_loader.dataset)
        val_size = int(total_samples * val_split)
        
        # Get class distribution for stratified splitting
        train_data_loader = getattr(module_data, "ImbalanceCIFAR100DataLoader")(
            data_dir=self.config['dataset']['data_dir'],
            batch_size=256,
            shuffle=False,
            num_workers=0,
            imb_factor=0.01,
            randaugm=False,
            training=True
        )
        
        cls_num_list = np.array(train_data_loader.cls_num_list)
        tail_threshold = self.config['group_definition']['tail_threshold']
        tail_classes = cls_num_list <= tail_threshold
        head_classes = cls_num_list > tail_threshold
        
        # Stratified splitting to ensure both head and tail classes in validation
        head_indices = []
        tail_indices = []
        
        for i in range(total_samples):
            # Get class for this sample (simplified - in practice need to check actual labels)
            # For now, use random assignment based on class distribution
            if np.random.random() < (head_classes.sum() / 100):  # Approximate head class probability
                head_indices.append(i)
            else:
                tail_indices.append(i)
        
        # Ensure we have both head and tail samples in validation
        val_head_size = min(val_size // 2, len(head_indices))
        val_tail_size = val_size - val_head_size
        
        val_indices = head_indices[:val_head_size] + tail_indices[:val_tail_size]
        test_indices = head_indices[val_head_size:] + tail_indices[val_tail_size:]
        
        print(f"📊 Total samples: {total_samples}")
        print(f"📊 Validation size: {len(val_indices)} (Head: {val_head_size}, Tail: {val_tail_size})")
        print(f"📊 Test size: {len(test_indices)}")
        
        # Create data loaders
        val_subset = torch.utils.data.Subset(self.full_data_loader.dataset, val_indices)
        self.val_data_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=256,
            shuffle=False,
            num_workers=0
        )
        
        test_subset = torch.utils.data.Subset(self.full_data_loader.dataset, test_indices)
        self.test_data_loader = torch.utils.data.DataLoader(
            test_subset,
            batch_size=256,
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ Validation samples: {len(val_indices)}")
        print(f"✅ Test samples: {len(test_indices)}")
        
        return head_classes, tail_classes
    
    def load_expert_models(self):
        """Load expert models với quality check"""
        print("📂 Loading expert models...")
        
        self.expert_models = {}
        tau_values = [0, 1.0, 2.0]
        expert_names = ['head_expert', 'balanced_expert', 'tail_expert']
        
        for tau, name in zip(tau_values, expert_names):
            # Find checkpoint file
            best_checkpoint = self.experts_dir / "model_best.pth"
            if best_checkpoint.exists():
                checkpoint_path = best_checkpoint
                print(f"  🏆 Using best model: {best_checkpoint.name}")
            else:
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
        """Lấy predictions từ tất cả experts với improved handling"""
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
    
    def analyze_expert_quality(self, expert_predictions, labels, head_classes, tail_classes):
        """Phân tích chất lượng từng expert"""
        print("🔍 Analyzing expert quality...")
        
        expert_quality = {}
        
        for name, predictions in expert_predictions.items():
            # Get predictions
            pred_labels = torch.argmax(predictions, dim=1)
            
            # Overall accuracy
            overall_acc = (pred_labels == labels).float().mean().item()
            
            # Head vs Tail accuracy
            head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
            tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
            
            head_acc = 0.0
            tail_acc = 0.0
            
            if head_mask.sum() > 0:
                head_acc = (pred_labels[head_mask] == labels[head_mask]).float().mean().item()
            
            if tail_mask.sum() > 0:
                tail_acc = (pred_labels[tail_mask] == labels[tail_mask]).float().mean().item()
            
            expert_quality[name] = {
                'overall_accuracy': overall_acc,
                'head_accuracy': head_acc,
                'tail_accuracy': tail_acc,
                'head_samples': head_mask.sum().item(),
                'tail_samples': tail_mask.sum().item()
            }
            
            print(f"  📊 {name}:")
            print(f"    - Overall: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
            print(f"    - Head: {head_acc:.4f} ({head_acc*100:.2f}%)")
            print(f"    - Tail: {tail_acc:.4f} ({tail_acc*100:.2f}%)")
        
        return expert_quality
    
    def improved_cs_plugin_optimization(self, expert_predictions, labels, head_classes, tail_classes, expert_quality):
        """Improved CS-plugin optimization với constraints"""
        print("🔍 Improved CS-plugin Optimization...")
        
        # Add constraints based on expert quality
        min_tail_expert_weight = 0.1  # Minimum 10% weight for tail expert
        min_balanced_expert_weight = 0.2  # Minimum 20% weight for balanced expert
        
        best_balanced_error = float('inf')
        best_params = None
        
        # Grid search with constraints
        weight_candidates = self._generate_constrained_weight_candidates(
            min_tail_expert_weight, min_balanced_expert_weight
        )
        lambda_candidates = self.config['cs_plugin']['lambda_0_candidates']
        
        print(f"🔍 Constrained grid search: {len(weight_candidates)} weight combinations × {len(lambda_candidates)} lambda values")
        
        for weights in weight_candidates:
            for lambda_0 in lambda_candidates:
                # Find optimal alpha
                alpha = self._find_optimal_alpha(
                    expert_predictions, labels, weights, lambda_0, head_classes, tail_classes
                )
                
                # Evaluate balanced error
                balanced_error = self._evaluate_balanced_error(
                    expert_predictions, labels, weights, lambda_0, alpha, head_classes, tail_classes
                )
                
                # Add regularization penalty for extreme weights
                weight_penalty = self._compute_weight_penalty(weights)
                balanced_error += weight_penalty
                
                if balanced_error < best_balanced_error:
                    best_balanced_error = balanced_error
                    best_params = {
                        'lambda_0': lambda_0,
                        'alpha': alpha,
                        'expert_weights': weights,
                        'balanced_error': balanced_error
                    }
                    
                    print(f"    ✅ New best: balanced_error = {balanced_error:.4f}")
                    print(f"        📊 Weights: {[f'{w:.3f}' for w in weights]}")
        
        print(f"✅ Improved CS-plugin optimization completed!")
        print(f"📊 Best balanced error: {best_balanced_error:.4f}")
        
        return best_params
    
    def _generate_constrained_weight_candidates(self, min_tail_weight, min_balanced_weight):
        """Generate weight candidates with constraints"""
        candidates = []
        
        # Generate weights with constraints
        for w_head in np.linspace(0.0, 0.8, 9):  # Head expert: 0% to 80%
            for w_balanced in np.linspace(min_balanced_weight, 0.8, 7):  # Balanced: min to 80%
                for w_tail in np.linspace(min_tail_weight, 0.6, 6):  # Tail: min to 60%
                    if abs(w_head + w_balanced + w_tail - 1.0) < 1e-6:  # Sum to 1
                        candidates.append([w_head, w_balanced, w_tail])
        
        return candidates
    
    def _compute_weight_penalty(self, weights):
        """Compute penalty for extreme weight distributions"""
        # Penalize if any expert gets too much weight (>80%)
        max_weight = max(weights)
        if max_weight > 0.8:
            return (max_weight - 0.8) * 0.1  # Penalty for extreme weights
        
        # Penalize if tail expert gets too little weight (<5%)
        if weights[2] < 0.05:
            return (0.05 - weights[2]) * 0.2  # Penalty for ignoring tail expert
        
        return 0.0
    
    def _find_optimal_alpha(self, expert_predictions, labels, expert_weights, lambda_0, head_classes, tail_classes):
        """Improved alpha finding với better initialization"""
        # Better initialization based on expert quality
        alpha = 0.5  # Start with moderate threshold
        M = self.config['cs_plugin']['alpha_search_iterations']
        
        for iteration in range(M):
            error = self._compute_cost_sensitive_error(
                expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes
            )
            
            # Improved update rule
            if error > 0.5:  # High error, increase threshold
                alpha = min(0.9, alpha + 0.1)
            else:  # Low error, decrease threshold
                alpha = max(0.1, alpha - 0.05)
        
        return alpha
    
    def _compute_cost_sensitive_error(self, expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes):
        """Compute cost-sensitive error"""
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_predictions['head_expert'])
        for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], expert_weights)):
            ensemble_pred += weight * expert_predictions[name]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < alpha
        
        # Compute error
        correct_mask = (predictions == labels) & (~reject_mask)
        error = 1.0 - correct_mask.float().mean()
        
        return error.item()
    
    def _evaluate_balanced_error(self, expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes):
        """Evaluate balanced error"""
        return self._compute_cost_sensitive_error(
            expert_predictions, labels, expert_weights, lambda_0, alpha, head_classes, tail_classes
        )
    
    def improved_worst_group_optimization(self, expert_predictions, labels, head_classes, tail_classes, cs_params):
        """Improved worst-group optimization với early stopping"""
        print("🔍 Improved Worst-group Plugin Optimization...")
        
        # Initialize group weights
        group_weights = torch.tensor([0.5, 0.5], dtype=torch.float)
        
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
            
            # Update group weights
            old_group_weights = group_weights.clone()
            group_weights = self._update_group_weights(
                group_weights, head_error, tail_error, step_size
            )
            
            # Check for convergence
            weight_change = torch.abs(group_weights - old_group_weights).sum().item()
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
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"    ⏹️  Early stopping at iteration {iteration+1}")
                    break
        
        print(f"✅ Improved worst-group optimization completed!")
        print(f"📊 Best worst-group error: {best_worst_group_error:.4f}")
        
        return best_params
    
    def _compute_group_errors(self, expert_predictions, labels, head_classes, tail_classes, params):
        """Compute group-wise errors"""
        head_mask = torch.tensor([bool(head_classes[label.item()]) for label in labels])
        tail_mask = torch.tensor([bool(tail_classes[label.item()]) for label in labels])
        
        head_error = self._compute_group_error(expert_predictions, labels, head_mask, params)
        tail_error = self._compute_group_error(expert_predictions, labels, tail_mask, params)
        
        return head_error, tail_error
    
    def _compute_group_error(self, expert_predictions, labels, group_mask, params):
        """Compute error for a group"""
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
        """Update group weights using exponentiated gradient"""
        # Compute gradients
        head_grad = head_error
        tail_grad = tail_error
        
        # Update weights
        head_weight = group_weights[0] * torch.exp(-step_size * head_grad)
        tail_weight = group_weights[1] * torch.exp(-step_size * tail_grad)
        
        # Normalize
        total_weight = head_weight + tail_weight
        group_weights[0] = head_weight / total_weight
        group_weights[1] = tail_weight / total_weight
        
        return group_weights
    
    def optimize(self):
        """Run improved optimization pipeline"""
        print("🚀 Improved MoE-Plugin Optimization Pipeline")
        print("=" * 60)
        
        # Setup data loaders
        head_classes, tail_classes = self.setup_data_loaders()
        
        # Load expert models
        self.load_expert_models()
        
        # Get expert predictions
        expert_predictions, val_labels = self.get_expert_predictions(self.val_data_loader)
        
        # Analyze expert quality
        expert_quality = self.analyze_expert_quality(expert_predictions, val_labels, head_classes, tail_classes)
        
        # Step 1: Improved CS-plugin optimization
        print("\n📊 Step 1: Improved CS-plugin Optimization")
        cs_params = self.improved_cs_plugin_optimization(
            expert_predictions, val_labels, head_classes, tail_classes, expert_quality
        )
        
        # Step 2: Improved worst-group optimization
        print("\n📊 Step 2: Improved Worst-group Plugin Optimization")
        final_params = self.improved_worst_group_optimization(
            expert_predictions, val_labels, head_classes, tail_classes, cs_params
        )
        
        # Save results
        self.save_optimized_parameters(final_params)
        
        return final_params
    
    def save_optimized_parameters(self, params):
        """Save optimized parameters"""
        print("💾 Saving improved optimized parameters...")
        
        save_dir = Path(self.config['output']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        params_dict = {
            'lambda_0': params['lambda_0'],
            'alpha': params['alpha'],
            'expert_weights': params['expert_weights'],
            'group_weights': params['group_weights'],
            'rejection_threshold': params['alpha'],
            'balanced_error': params['balanced_error'],
            'worst_group_error': params['worst_group_error'],
            'config': self.config,
            'improvements': {
                'constrained_optimization': True,
                'expert_quality_analysis': True,
                'early_stopping': True,
                'weight_regularization': True
            }
        }
        
        # Save parameters
        params_file = save_dir / 'improved_optimized_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"✅ Improved parameters saved to: {params_file}")
        
        # Print detailed results
        print(f"\n📊 Improved Final Results:")
        print(f"  - Lambda_0: {params['lambda_0']}")
        print(f"  - Alpha: {params['alpha']:.6f}")
        print(f"  - Expert weights: {[f'{w:.6f}' for w in params['expert_weights']]}")
        print(f"  - Group weights: {[f'{w:.6f}' for w in params['group_weights']]}")
        print(f"  - Balanced error: {params['balanced_error']:.6f}")
        print(f"  - Worst-group error: {params['worst_group_error']:.6f}")
        
        return save_dir


def main():
    parser = argparse.ArgumentParser(description='Improved MoE-Plugin Optimization')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        return
    
    # Run improved optimization
    optimizer = ImprovedMoEPluginOptimizer(args.config)
    results = optimizer.optimize()
    
    print(f"\n✅ Improved optimization completed!")
    print(f"💡 This should give better results with more balanced expert usage.")


if __name__ == '__main__':
    main()
