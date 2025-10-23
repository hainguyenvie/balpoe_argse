#!/usr/bin/env python3
"""
Script để plot Risk-Coverage Curves theo paper evaluation metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_risk_coverage_results(results_file):
    """Load risk-coverage results từ JSON file"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def plot_risk_coverage_curves(results, save_dir):
    """Plot risk-coverage curves theo paper format"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    rejection_rates = np.array(results['rejection_rates'])
    balanced_errors = np.array(results['balanced_errors'])
    worst_group_errors = np.array(results['worst_group_errors'])
    
    # Plot 1: Balanced Error vs Rejection Rate
    ax1.plot(rejection_rates, balanced_errors, 'o-', linewidth=2, markersize=6, label='Balanced Error')
    ax1.set_xlabel('Rejection Rate', fontsize=12)
    ax1.set_ylabel('Balanced Error', fontsize=12)
    ax1.set_title(f'Balanced Error vs Rejection Rate\nAURC = {results["balanced_aurc"]:.4f}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Worst-Group Error vs Rejection Rate
    ax2.plot(rejection_rates, worst_group_errors, 's-', linewidth=2, markersize=6, label='Worst-Group Error', color='red')
    ax2.set_xlabel('Rejection Rate', fontsize=12)
    ax2.set_ylabel('Worst-Group Error', fontsize=12)
    ax2.set_title(f'Worst-Group Error vs Rejection Rate\nAURC = {results["worst_group_aurc"]:.4f}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_file = Path(save_dir) / 'risk_coverage_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Risk-coverage curves saved to {plot_file}")
    
    # Also save as PDF for publication
    pdf_file = Path(save_dir) / 'risk_coverage_curves.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"📊 Risk-coverage curves (PDF) saved to {pdf_file}")
    
    plt.show()
    
    return plot_file, pdf_file

def plot_combined_curves(results, save_dir):
    """Plot combined risk-coverage curves trên cùng một plot"""
    
    plt.figure(figsize=(10, 8))
    
    # Extract data
    rejection_rates = np.array(results['rejection_rates'])
    balanced_errors = np.array(results['balanced_errors'])
    worst_group_errors = np.array(results['worst_group_errors'])
    
    # Plot both curves
    plt.plot(rejection_rates, balanced_errors, 'o-', linewidth=2, markersize=6, 
             label=f'Balanced Error (AURC = {results["balanced_aurc"]:.4f})', color='blue')
    plt.plot(rejection_rates, worst_group_errors, 's-', linewidth=2, markersize=6, 
             label=f'Worst-Group Error (AURC = {results["worst_group_aurc"]:.4f})', color='red')
    
    plt.xlabel('Rejection Rate', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Risk-Coverage Curves: Balanced vs Worst-Group Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Save combined plot
    combined_file = Path(save_dir) / 'combined_risk_coverage_curves.png'
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"📊 Combined risk-coverage curves saved to {combined_file}")
    
    plt.show()
    
    return combined_file

def print_summary_table(results):
    """Print summary table của results"""
    
    print("\n" + "="*80)
    print("📊 RISK-COVERAGE CURVES SUMMARY")
    print("="*80)
    
    print(f"{'Rejection Cost':<15} {'Rejection Rate':<15} {'Balanced Error':<15} {'Worst-Group Error':<15}")
    print("-"*80)
    
    for i, cost in enumerate(results['rejection_costs']):
        print(f"{cost:<15.3f} {results['rejection_rates'][i]:<15.4f} {results['balanced_errors'][i]:<15.4f} {results['worst_group_errors'][i]:<15.4f}")
    
    print("-"*80)
    print(f"{'AURC':<15} {'':<15} {results['balanced_aurc']:<15.4f} {results['worst_group_aurc']:<15.4f}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Plot Risk-Coverage Curves')
    parser.add_argument('--results_file', type=str, 
                       default='checkpoints/plugin_optimized/risk_coverage_results.json',
                       help='Path to risk-coverage results JSON file')
    parser.add_argument('--save_dir', type=str, 
                       default='checkpoints/plugin_optimized',
                       help='Directory to save plots')
    parser.add_argument('--show', action='store_true', 
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Load results
    print(f"📂 Loading results from {args.results_file}")
    results = load_risk_coverage_results(args.results_file)
    
    # Print summary
    print_summary_table(results)
    
    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot individual curves
    print("\n📊 Plotting individual risk-coverage curves...")
    plot_risk_coverage_curves(results, args.save_dir)
    
    # Plot combined curves
    print("\n📊 Plotting combined risk-coverage curves...")
    plot_combined_curves(results, args.save_dir)
    
    print(f"\n✅ All plots saved to {args.save_dir}")

if __name__ == '__main__':
    main()
