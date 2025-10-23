#!/usr/bin/env python3
"""
Debug script để kiểm tra chất lượng từng expert riêng lẻ
Giúp hiểu tại sao tail expert bị bỏ qua trong optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import data_loader.data_loaders as module_data
import model.model as module_arch


def load_expert_model(expert_name, experts_dir, device='cpu'):
    """Load một expert riêng lẻ"""
    print(f"📂 Loading {expert_name}...")
    
    # Find checkpoint file
    best_checkpoint = experts_dir / "model_best.pth"
    if best_checkpoint.exists():
        checkpoint_path = best_checkpoint
        print(f"  🏆 Using best model: {best_checkpoint.name}")
    else:
        checkpoint_files = list(experts_dir.glob(f"*epoch*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {experts_dir}")
        checkpoint_path = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        print(f"  ⚠️  Best model not found, using latest: {checkpoint_path.name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    return model


def get_expert_predictions_single(model, data_loader, expert_name, device='cpu'):
    """Lấy predictions từ một expert riêng lẻ"""
    print(f"🔍 Getting predictions from {expert_name}...")
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            
            # Get expert prediction
            expert_output = model(batch_data)
            
            # Handle BalPoE model output (can be dict or tensor)
            if isinstance(expert_output, dict):
                # BalPoE returns dict with 'logits' key containing individual expert logits
                if 'logits' in expert_output:
                    # Get individual expert logits (shape: [batch_size, num_experts, num_classes])
                    all_expert_logits = expert_output['logits']
                    # Extract specific expert based on name
                    expert_idx = {'head_expert': 0, 'balanced_expert': 1, 'tail_expert': 2}[expert_name]
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
            
            all_predictions.append(expert_probs.cpu())
            all_labels.append(batch_labels)
    
    # Combine predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"✅ {expert_name} predictions shape: {all_predictions.shape}")
    print(f"✅ Labels shape: {all_labels.shape}")
    
    return all_predictions, all_labels


def compute_expert_metrics(predictions, labels, head_classes, tail_classes):
    """Tính các metrics cho một expert"""
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
    
    # Confidence analysis
    max_probs = torch.max(predictions, dim=1)[0]
    avg_confidence = max_probs.mean().item()
    
    # Prediction diversity (entropy)
    entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
    avg_entropy = entropy.mean().item()
    
    return {
        'overall_accuracy': overall_acc,
        'head_accuracy': head_acc,
        'tail_accuracy': tail_acc,
        'avg_confidence': avg_confidence,
        'avg_entropy': avg_entropy,
        'head_samples': head_mask.sum().item(),
        'tail_samples': tail_mask.sum().item()
    }


def analyze_expert_quality(experts_dir, config_path):
    """Phân tích chất lượng từng expert"""
    print("🔍 Analyzing Expert Quality...")
    print("=" * 60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup data loader
    data_loader_config = {
        "type": "ImbalanceCIFAR100DataLoader",
        "args": {
            "data_dir": config['dataset']['data_dir'],
            "batch_size": 256,
            "shuffle": False,
            "num_workers": 0,  # Avoid multiprocessing issues
            "imb_factor": 0.01,
            "randaugm": False,
            "training": False
        }
    }
    
    # Create data loader
    data_loader = getattr(module_data, data_loader_config['type'])(
        **data_loader_config['args']
    )
    
    # Get class distribution for group definition
    train_data_loader = getattr(module_data, "ImbalanceCIFAR100DataLoader")(
        data_dir=config['dataset']['data_dir'],
        batch_size=256,
        shuffle=False,
        num_workers=0,
        imb_factor=0.01,
        randaugm=False,
        training=True
    )
    
    cls_num_list = np.array(train_data_loader.cls_num_list)
    tail_threshold = config['group_definition']['tail_threshold']
    tail_classes = cls_num_list <= tail_threshold
    head_classes = cls_num_list > tail_threshold
    
    print(f"📊 Dataset info:")
    print(f"  - Total samples: {len(data_loader.sampler)}")
    print(f"  - Head classes: {head_classes.sum()} (samples > {tail_threshold})")
    print(f"  - Tail classes: {tail_classes.sum()} (samples <= {tail_threshold})")
    
    # Analyze each expert
    expert_names = ['head_expert', 'balanced_expert', 'tail_expert']
    expert_taus = [0, 1.0, 2.0]
    
    results = {}
    
    for name, tau in zip(expert_names, expert_taus):
        print(f"\n{'='*20} {name.upper()} (τ={tau}) {'='*20}")
        
        # Load expert model
        model = load_expert_model(name, Path(experts_dir))
        
        # Get predictions
        predictions, labels = get_expert_predictions_single(model, data_loader, name)
        
        # Compute metrics
        metrics = compute_expert_metrics(predictions, labels, head_classes, tail_classes)
        
        # Print results
        print(f"📊 {name} Performance:")
        print(f"  - Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        print(f"  - Head Accuracy: {metrics['head_accuracy']:.4f} ({metrics['head_accuracy']*100:.2f}%)")
        print(f"  - Tail Accuracy: {metrics['tail_accuracy']:.4f} ({metrics['tail_accuracy']*100:.2f}%)")
        print(f"  - Avg Confidence: {metrics['avg_confidence']:.4f}")
        print(f"  - Avg Entropy: {metrics['avg_entropy']:.4f}")
        print(f"  - Head Samples: {metrics['head_samples']}")
        print(f"  - Tail Samples: {metrics['tail_samples']}")
        
        # Analyze prediction patterns
        pred_labels = torch.argmax(predictions, dim=1)
        max_probs = torch.max(predictions, dim=1)[0]
        
        print(f"📊 {name} Prediction Patterns:")
        print(f"  - Min confidence: {max_probs.min():.4f}")
        print(f"  - Max confidence: {max_probs.max():.4f}")
        print(f"  - Std confidence: {max_probs.std():.4f}")
        
        # Check if expert is biased towards certain classes
        class_counts = torch.bincount(pred_labels, minlength=100)
        most_predicted = torch.argmax(class_counts).item()
        most_predicted_count = class_counts[most_predicted].item()
        total_predictions = class_counts.sum().item()
        
        print(f"  - Most predicted class: {most_predicted} ({most_predicted_count}/{total_predictions} = {most_predicted_count/total_predictions:.2%})")
        
        results[name] = metrics
    
    # Compare experts
    print(f"\n{'='*20} EXPERT COMPARISON {'='*20}")
    print(f"{'Expert':<15} {'Overall':<8} {'Head':<8} {'Tail':<8} {'Conf':<8} {'Entropy':<8}")
    print("-" * 65)
    
    for name in expert_names:
        m = results[name]
        print(f"{name:<15} {m['overall_accuracy']:<8.3f} {m['head_accuracy']:<8.3f} {m['tail_accuracy']:<8.3f} {m['avg_confidence']:<8.3f} {m['avg_entropy']:<8.3f}")
    
    # Identify issues
    print(f"\n🔍 Analysis:")
    
    # Check if tail expert is actually good at tail classes
    tail_expert_tail_acc = results['tail_expert']['tail_accuracy']
    balanced_expert_tail_acc = results['balanced_expert']['tail_accuracy']
    head_expert_tail_acc = results['head_expert']['tail_accuracy']
    
    print(f"  - Tail expert tail accuracy: {tail_expert_tail_acc:.4f}")
    print(f"  - Balanced expert tail accuracy: {balanced_expert_tail_acc:.4f}")
    print(f"  - Head expert tail accuracy: {head_expert_tail_acc:.4f}")
    
    if tail_expert_tail_acc < balanced_expert_tail_acc:
        print(f"  ⚠️  WARNING: Tail expert is WORSE than balanced expert on tail classes!")
        print(f"      This explains why it gets 0% weight in optimization.")
    else:
        print(f"  ✅ Tail expert is better than balanced expert on tail classes.")
    
    # Check overall performance
    tail_expert_overall = results['tail_expert']['overall_accuracy']
    balanced_expert_overall = results['balanced_expert']['overall_accuracy']
    
    if tail_expert_overall < balanced_expert_overall:
        print(f"  ⚠️  WARNING: Tail expert has lower overall accuracy than balanced expert.")
    else:
        print(f"  ✅ Tail expert has competitive overall accuracy.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Debug Expert Quality')
    parser.add_argument('--experts_dir', type=str, required=True,
                       help='Path to experts directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not Path(args.experts_dir).exists():
        print(f"❌ Experts directory not found: {args.experts_dir}")
        return
    
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        return
    
    # Run analysis
    results = analyze_expert_quality(args.experts_dir, args.config)
    
    print(f"\n✅ Expert quality analysis completed!")
    print(f"💡 This analysis helps explain why certain experts get low weights in optimization.")


if __name__ == '__main__':
    main()
