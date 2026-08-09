"""
Main script to run regression training or evaluation.

Usage:
    python run_regression.py --mode train --config config_train.yaml
    python run_regression.py --mode test --config config_test.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml

from regression_trainer import RegressionTrainer
from regression_evaluator import RegressionEvaluator

import numpy as np  # For summary stats

# ── Path anchors ──────────────────────────────────────────────────────────────
# Config files store paths relative to the iml-dl repo root so that nothing in
# this project hardcodes an absolute location. They are resolved here, against
# the repo root derived from this file, so runs work from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]

_PATH_KEYS = (
    'regression_gt_train_location',
    'regression_gt_val_location',
    'regression_gt_test_location',
    'regress_model_dir',
)


def resolve_config_paths(config):
    """Resolve relative path entries in a config against the iml-dl repo root.

    Absolute paths are left untouched, so existing absolute configs keep working.
    """
    rp = config.get('regression_params')
    if not isinstance(rp, dict):
        return config
    for key in _PATH_KEYS:
        val = rp.get(key)
        if isinstance(val, str) and val and not os.path.isabs(val):
            resolved = (REPO_ROOT / val).resolve()
            # preserve a trailing separator, which some loaders rely on
            rp[key] = str(resolved) + ('/' if val.endswith('/') else '')
    return config

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('regression.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path):
    """Load configuration from YAML file, resolving repo-relative data paths."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return resolve_config_paths(config)


def train_regression(config):
    """Train regression model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Initialize trainer
    trainer = RegressionTrainer(config, device)
    
    # Run training
    logging.info('Starting regression training...')
    best_metrics = trainer.train()
    
    logging.info('Training completed!')
    logging.info(f'Best validation T2 MAE: {best_metrics["mae_t2"]:.6f}')
    
    return best_metrics


def test_regression(config):
    """Test regression model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Initialize evaluator
    evaluator = RegressionEvaluator(config, device)
    
    # Run evaluation
    logging.info('Starting regression evaluation...')
    metrics = evaluator.evaluate()
    
    logging.info('Evaluation completed!')
    
# 1. Main Overall Metrics Printing
    print("\n" + "="*30)
    print("OVERALL PARAMETER METRICS")
    print("="*30)
    for param_type in ['t2', 's0', 'recon_t2', 'recon_s0', 'magnitude']:
        if param_type in metrics and metrics[param_type]['nrmse']:
            param_name = {
                't2': 'T2* (Model Prediction)',
                's0': 'S0 (Model Prediction)',
                'recon_t2': 'T2* (Baseline - Signal Model)',
                'recon_s0': 'S0 (Baseline - Signal Model)',
                'magnitude': 'Magnitude Reconstruction'
            }[param_type]
            
            print(f'\n{param_name}:')
            print(f'  NRMSE: {np.mean(metrics[param_type]["nrmse"]):.6f} ± {np.std(metrics[param_type]["nrmse"]):.6f}')
            print(f'  SSIM:  {np.mean(metrics[param_type]["ssim"]):.4f} ± {np.std(metrics[param_type]["ssim"]):.4f}')
            print(f'  PSNR:  {np.mean(metrics[param_type]["psnr"]):.2f} ± {np.std(metrics[param_type]["psnr"]):.2f} dB')

    # 2. Tissue-Specific T2* Metrics Printing
    print("\n" + "="*30)
    print("TISSUE-SPECIFIC T2* ANALYSIS")
    print("="*30)
    
    tissue_map = {'wm': 'White Matter', 'gm': 'Gray Matter', 'csf': 'CSF'}
    
    for tissue, name in tissue_map.items():
        pred_key = f'{tissue}_t2'
        base_key = f'{tissue}_recon_t2'
        
        if pred_key in metrics and metrics[pred_key]['nrmse']:
            print(f'\n[{name}]')
            
            # Extract values
            p_nrmse = metrics[pred_key]['nrmse']
            b_nrmse = metrics[base_key]['nrmse']
            
            # Calculate means/stds
            m_p_nrmse, s_p_nrmse = np.mean(p_nrmse), np.std(p_nrmse)
            m_b_nrmse, s_b_nrmse = np.mean(b_nrmse), np.std(b_nrmse)
            
            # Print Comparison
            print(f'  Prediction NRMSE: {m_p_nrmse:.6f} ± {s_p_nrmse:.6f}')
            print(f'  Baseline NRMSE:   {m_b_nrmse:.6f} ± {s_b_nrmse:.6f}')
            
            # Calculate and print improvement
            improvement = ((m_b_nrmse - m_p_nrmse) / m_b_nrmse) * 100
            print(f'  NRMSE Improvement: {improvement:+.2f}%')

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run regression training or testing')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    logging.info(f'Loading configuration from: {args.config}')
    config = load_config(args.config)
    
    # Run appropriate mode
    if args.mode == 'train':
        train_regression(config)
    elif args.mode == 'test':
        test_regression(config)


if __name__ == '__main__':
    main()