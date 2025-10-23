#!/usr/bin/env python3
"""
Gating Network để học trọng số cho các expert
Thay thế grid search bằng mạng neural network có khả năng học
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class GatingNetwork(nn.Module):
    """
    Mạng Gating để học trọng số cho các expert
    Architecture: MLP với input là feature vector, output là expert weights
    """
    
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = [256, 128], 
                 num_experts: int = 3, dropout_rate: float = 0.5):
        """
        Args:
            input_dim: Kích thước vector đặc trưng (512 cho ResNet-32)
            hidden_dims: Kích thước các lớp ẩn
            num_experts: Số lượng experts (3: head, balanced, tail)
            dropout_rate: Tỷ lệ dropout
        """
        super(GatingNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer: 3 neurons với softmax
        layers.append(nn.Linear(prev_dim, num_experts))
        
        self.network = nn.Sequential(*layers)
        
        print(f"🏗️  Gating Network Architecture:")
        print(f"   Input: {input_dim} (feature vector)")
        print(f"   Hidden: {hidden_dims}")
        print(f"   Output: {num_experts} (expert weights)")
        print(f"   Dropout: {dropout_rate}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            features: [batch_size, input_dim] - feature vectors
        Returns:
            weights: [batch_size, num_experts] - expert weights (softmax)
        """
        # Get raw logits
        logits = self.network(features)
        
        # Apply softmax to get normalized weights
        weights = F.softmax(logits, dim=1)
        
        return weights


class GatingNetworkTrainer:
    """
    Trainer cho Gating Network
    Sử dụng Cost-Sensitive Cross-Entropy Loss
    """
    
    def __init__(self, gating_network: GatingNetwork, config: Dict):
        self.gating_network = gating_network
        self.config = config
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_epochs = config.get('num_epochs', 100)
        self.batch_size = config.get('batch_size', 256)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.gating_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )
        
        print(f"🎯 Gating Network Trainer:")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Weight decay: {self.weight_decay}")
    
    def compute_class_weights(self, labels: torch.Tensor, num_classes: int = 100) -> torch.Tensor:
        """
        Tính class weights cho Cost-Sensitive Loss
        β_y ∝ 1/N_y với N_y là số samples của class y
        """
        # Count samples per class
        class_counts = torch.bincount(labels, minlength=num_classes).float()
        
        # Avoid division by zero
        class_counts = torch.clamp(class_counts, min=1.0)
        
        # Compute weights: β_y = 1/N_y
        class_weights = 1.0 / class_counts
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_classes
        
        return class_weights
    
    def compute_cost_sensitive_loss(self, mixed_probs: torch.Tensor, labels: torch.Tensor, 
                                   class_weights: torch.Tensor) -> torch.Tensor:
        """
        Tính Cost-Sensitive Cross-Entropy Loss
        L_gating = Σ_i Σ_y β_y · I(y_i = y) · log(p_mix(y|x_i))
        """
        # Get class weights for current batch
        batch_class_weights = class_weights[labels]
        
        # Compute log probabilities
        log_probs = torch.log(mixed_probs + 1e-8)  # Add small epsilon for numerical stability
        
        # Get log probabilities for true labels
        true_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Compute weighted loss
        weighted_loss = -batch_class_weights * true_log_probs
        
        return weighted_loss.mean()
    
    def train_epoch(self, data_loader: DataLoader, expert_models: Dict, 
                   class_weights: torch.Tensor, device: str = 'cpu') -> float:
        """
        Train một epoch
        """
        self.gating_network.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Get expert predictions (frozen)
            expert_predictions = {}
            with torch.no_grad():
                for name, model in expert_models.items():
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
            
            # Get feature vectors (use balanced expert features)
            with torch.no_grad():
                balanced_output = expert_models['balanced_expert'](batch_data)
                if isinstance(balanced_output, dict) and 'feat' in balanced_output:
                    features = balanced_output['feat']
                else:
                    # Fallback: try to get features from model
                    try:
                        features = expert_models['balanced_expert'].get_features(batch_data)
                    except:
                        # If no get_features method, use the last layer before classifier
                        features = balanced_output['output'] if isinstance(balanced_output, dict) else balanced_output
                
                # Ensure features are 2D
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
            
            # Forward pass through gating network
            expert_weights = self.gating_network(features)
            
            # Compute mixed probabilities
            mixed_probs = torch.zeros_like(expert_predictions['head_expert'])
            for i, (name, weight) in enumerate(zip(['head_expert', 'balanced_expert', 'tail_expert'], 
                                                 expert_weights.T)):
                mixed_probs += weight.unsqueeze(1) * expert_predictions[name]
            
            # Compute loss
            loss = self.compute_cost_sensitive_loss(mixed_probs, batch_labels, class_weights)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, data_loader: DataLoader, expert_models: Dict, 
              labels: torch.Tensor, device: str = 'cpu') -> Dict:
        """
        Train gating network
        """
        print("🚀 Bắt đầu training Gating Network...")
        
        # Compute class weights
        class_weights = self.compute_class_weights(labels)
        print(f"📊 Class weights computed (min: {class_weights.min():.4f}, max: {class_weights.max():.4f})")
        
        # Training loop
        train_losses = []
        best_loss = float('inf')
        best_state_dict = None
        
        for epoch in range(self.num_epochs):
            # Train one epoch
            avg_loss = self.train_epoch(data_loader, expert_models, class_weights, device)
            train_losses.append(avg_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state_dict = self.gating_network.state_dict().copy()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{self.num_epochs}: Loss={avg_loss:.4f}, LR={current_lr:.6f}")
        
        # Load best model
        if best_state_dict is not None:
            self.gating_network.load_state_dict(best_state_dict)
            print(f"✅ Loaded best model with loss: {best_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'best_loss': best_loss,
            'final_loss': train_losses[-1]
        }
    
    def save_model(self, save_path: Path):
        """Save gating network"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.gating_network.state_dict(), save_path / 'gating_network.pth')
        
        # Save training config
        config_path = save_path / 'gating_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"💾 Gating network saved to: {save_path}")
    
    def load_model(self, load_path: Path):
        """Load gating network"""
        model_path = load_path / 'gating_network.pth'
        if model_path.exists():
            self.gating_network.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"📂 Gating network loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")


def create_gating_network(input_dim: int = 512, config: Optional[Dict] = None) -> GatingNetwork:
    """
    Factory function để tạo Gating Network
    """
    if config is None:
        config = {
            'hidden_dims': [256, 128],
            'num_experts': 3,
            'dropout_rate': 0.5
        }
    
    gating_network = GatingNetwork(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        num_experts=config['num_experts'],
        dropout_rate=config['dropout_rate']
    )
    
    return gating_network


def create_gating_trainer(gating_network: GatingNetwork, config: Optional[Dict] = None) -> GatingNetworkTrainer:
    """
    Factory function để tạo Gating Network Trainer
    """
    if config is None:
        config = {
            'learning_rate': 0.001,
            'num_epochs': 100,
            'batch_size': 256,
            'weight_decay': 1e-4
        }
    
    trainer = GatingNetworkTrainer(gating_network, config)
    return trainer
