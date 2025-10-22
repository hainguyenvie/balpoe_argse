"""
MoE-Plugin Evaluator
Đánh giá và so sánh MoE-Plugin với các baselines
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
class EvaluationResults:
    """Kết quả đánh giá"""
    method_name: str
    overall_accuracy: float
    balanced_accuracy: float
    worst_group_accuracy: float
    many_shot_accuracy: float
    medium_shot_accuracy: float
    few_shot_accuracy: float
    rejection_rate: float
    head_group_accuracy: float
    tail_group_accuracy: float


class MoEPluginModel:
    """MoE-Plugin Model với rejection capability"""
    
    def __init__(self, expert_ensemble, plugin_params):
        self.expert_ensemble = expert_ensemble
        self.params = plugin_params
        
    def predict_with_rejection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dự đoán với khả năng từ chối"""
        
        # Get expert predictions
        expert_probs = self.expert_ensemble.predict_probs(x)
        
        # Weighted ensemble prediction
        ensemble_pred = torch.zeros_like(expert_probs[:, 0, :])
        for i, weight in enumerate(self.params.expert_weights):
            ensemble_pred += weight * expert_probs[:, i, :]
        
        # Apply rejection rule
        max_probs, predictions = torch.max(ensemble_pred, dim=1)
        reject_mask = max_probs < self.params.alpha
        
        return predictions, reject_mask
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Dự đoán không từ chối (cho so sánh)"""
        predictions, _ = self.predict_with_rejection(x)
        return predictions


class BalPoEBaseline:
    """BalPoE Baseline - trung bình logits của các experts"""
    
    def __init__(self, expert_ensemble):
        self.expert_ensemble = expert_ensemble
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Dự đoán bằng cách lấy trung bình logits"""
        
        # Get expert predictions
        expert_probs = self.expert_ensemble.predict_probs(x)
        
        # Average ensemble prediction
        ensemble_pred = expert_probs.mean(dim=1)
        
        # Get predictions
        _, predictions = torch.max(ensemble_pred, dim=1)
        
        return predictions


class ChowsRuleBaseline:
    """Chow's Rule Baseline - single expert với rejection rule"""
    
    def __init__(self, expert_ensemble, rejection_threshold=0.5):
        self.expert_ensemble = expert_ensemble
        self.rejection_threshold = rejection_threshold
        
    def predict_with_rejection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dự đoán với Chow's rule rejection"""
        
        # Use balanced expert (middle expert)
        balanced_expert = list(self.expert_ensemble.experts.values())[1]  # balanced expert
        
        with torch.no_grad():
            expert_output = balanced_expert(x)
            expert_probs = F.softmax(expert_output, dim=1)
        
        # Apply Chow's rule
        max_probs, predictions = torch.max(expert_probs, dim=1)
        reject_mask = max_probs < self.rejection_threshold
        
        return predictions, reject_mask
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Dự đoán không từ chối"""
        predictions, _ = self.predict_with_rejection(x)
        return predictions


class MoEPluginEvaluator:
    """Evaluator cho MoE-Plugin và các baselines"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_model(self, model, data_loader, labels: torch.Tensor, 
                      model_name: str, use_rejection: bool = False) -> EvaluationResults:
        """Đánh giá một model"""
        
        print(f"📊 Evaluating {model_name}...")
        
        model.eval()
        all_predictions = []
        all_rejections = []
        
        with torch.no_grad():
            for batch_data, _ in data_loader:
                batch_data = batch_data.to(self.device)
                
                if use_rejection and hasattr(model, 'predict_with_rejection'):
                    batch_predictions, batch_rejections = model.predict_with_rejection(batch_data)
                    all_rejections.append(batch_rejections.cpu())
                else:
                    batch_predictions = model.predict(batch_data)
                    all_rejections.append(torch.zeros(batch_predictions.size(0), dtype=torch.bool))
                
                all_predictions.append(batch_predictions.cpu())
        
        # Combine results
        predictions = torch.cat(all_predictions, dim=0)
        rejections = torch.cat(all_rejections, dim=0)
        
        # Compute metrics
        results = self._compute_metrics(predictions, labels, rejections, model_name)
        
        return results
    
    def _compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor, 
                        rejections: torch.Tensor, model_name: str) -> EvaluationResults:
        """Tính toán các metrics"""
        
        # Overall accuracy
        correct_mask = (predictions == labels) & (~rejections)
        overall_accuracy = correct_mask.float().mean().item()
        
        # Rejection rate
        rejection_rate = rejections.float().mean().item()
        
        # Group-wise accuracy (simplified grouping)
        head_mask = labels < 50  # First 50 classes as head
        tail_mask = labels >= 50  # Last 50 classes as tail
        
        head_correct = ((predictions == labels) & (~rejections) & head_mask).float().sum()
        head_total = head_mask.float().sum()
        head_accuracy = (head_correct / head_total).item() if head_total > 0 else 0.0
        
        tail_correct = ((predictions == labels) & (~rejections) & tail_mask).float().sum()
        tail_total = tail_mask.float().sum()
        tail_accuracy = (tail_correct / tail_total).item() if tail_total > 0 else 0.0
        
        # Balanced accuracy
        balanced_accuracy = (head_accuracy + tail_accuracy) / 2.0
        
        # Worst-group accuracy
        worst_group_accuracy = min(head_accuracy, tail_accuracy)
        
        # Many/Medium/Few shot accuracy (simplified)
        # Cần implement logic phân chia dựa trên số lượng samples thực tế
        many_shot_accuracy = head_accuracy  # Simplified
        medium_shot_accuracy = (head_accuracy + tail_accuracy) / 2.0  # Simplified
        few_shot_accuracy = tail_accuracy  # Simplified
        
        return EvaluationResults(
            method_name=model_name,
            overall_accuracy=overall_accuracy,
            balanced_accuracy=balanced_accuracy,
            worst_group_accuracy=worst_group_accuracy,
            many_shot_accuracy=many_shot_accuracy,
            medium_shot_accuracy=medium_shot_accuracy,
            few_shot_accuracy=few_shot_accuracy,
            rejection_rate=rejection_rate,
            head_group_accuracy=head_accuracy,
            tail_group_accuracy=tail_accuracy
        )
    
    def compare_methods(self, models: Dict[str, any], data_loader, labels: torch.Tensor) -> List[EvaluationResults]:
        """So sánh nhiều methods"""
        
        print("🔍 Comparing methods...")
        
        results = []
        
        for model_name, model in models.items():
            use_rejection = 'plugin' in model_name.lower() or 'chow' in model_name.lower()
            result = self.evaluate_model(model, data_loader, labels, model_name, use_rejection)
            results.append(result)
            
            print(f"✅ {model_name}:")
            print(f"  - Overall Accuracy: {result.overall_accuracy:.4f}")
            print(f"  - Balanced Accuracy: {result.balanced_accuracy:.4f}")
            print(f"  - Worst-group Accuracy: {result.worst_group_accuracy:.4f}")
            print(f"  - Rejection Rate: {result.rejection_rate:.4f}")
            print()
        
        return results
    
    def print_comparison_table(self, results: List[EvaluationResults]):
        """In bảng so sánh"""
        
        print("📊 COMPARISON RESULTS")
        print("=" * 100)
        
        # Header
        print(f"{'Method':<20} {'Overall':<8} {'Balanced':<9} {'Worst-Group':<12} {'Rejection':<9} {'Head':<6} {'Tail':<6}")
        print("-" * 100)
        
        # Results
        for result in results:
            print(f"{result.method_name:<20} "
                  f"{result.overall_accuracy:<8.4f} "
                  f"{result.balanced_accuracy:<9.4f} "
                  f"{result.worst_group_accuracy:<12.4f} "
                  f"{result.rejection_rate:<9.4f} "
                  f"{result.head_group_accuracy:<6.4f} "
                  f"{result.tail_group_accuracy:<6.4f}")
        
        print("=" * 100)
    
    def save_results(self, results: List[EvaluationResults], save_path: str):
        """Lưu kết quả"""
        
        results_dict = []
        for result in results:
            results_dict.append({
                'method_name': result.method_name,
                'overall_accuracy': result.overall_accuracy,
                'balanced_accuracy': result.balanced_accuracy,
                'worst_group_accuracy': result.worst_group_accuracy,
                'many_shot_accuracy': result.many_shot_accuracy,
                'medium_shot_accuracy': result.medium_shot_accuracy,
                'few_shot_accuracy': result.few_shot_accuracy,
                'rejection_rate': result.rejection_rate,
                'head_group_accuracy': result.head_group_accuracy,
                'tail_group_accuracy': result.tail_group_accuracy
            })
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"💾 Results saved to: {save_path}")
