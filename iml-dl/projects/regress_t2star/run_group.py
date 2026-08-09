"""
Run all YAML configs in a specified directory
Simple and straightforward!
"""

import argparse
import logging
from pathlib import Path
import time
from datetime import datetime

# Import your existing functions
from run_regression import setup_logging, load_config, train_regression


def run_all_configs_in_directory(config_dir):
    """
    Run all YAML files in the specified directory.
    
    Args:
        config_dir: Path to directory containing YAML configs
    """
    config_path = Path(config_dir)
    
    # Check directory exists
    if not config_path.exists():
        logging.error(f"Directory not found: {config_dir}")
        return
    
    # Find all YAML files
    yaml_files = sorted(config_path.glob('*.yaml'))
    
    if not yaml_files:
        logging.error(f"No YAML files found in {config_dir}")
        return
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Found {len(yaml_files)} experiment configs in:")
    logging.info(f"{config_dir}")
    logging.info(f"{'='*60}\n")
    
    # Track results
    successful = 0
    failed = 0
    results = []
    
    start_time = time.time()
    
    # Run each config
    for i, yaml_file in enumerate(yaml_files, 1):
        logging.info(f"\n[{i}/{len(yaml_files)}] Running: {yaml_file.name}")
        logging.info("-" * 60)
        
        exp_start = time.time()
        
        try:
            # Load and run
            config = load_config(str(yaml_file))
            metrics = train_regression(config)
            
            exp_time = time.time() - exp_start
            
            # Success
            successful += 1
            mae = metrics.get('mae_t2', 0.0)
            logging.info(f"✅ SUCCESS - MAE: {mae:.4f} - Time: {exp_time/60:.1f} min")
            results.append((yaml_file.name, True, mae, exp_time))
            
        except Exception as e:
            exp_time = time.time() - exp_start
            
            # Failure
            failed += 1
            logging.error(f"❌ FAILED - Error: {str(e)} - Time: {exp_time/60:.1f} min")
            results.append((yaml_file.name, False, None, exp_time))
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✅ Successful: {successful}/{len(yaml_files)}")
    print(f"❌ Failed: {failed}/{len(yaml_files)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    if successful > 0:
        print("\n--- Successful Experiments ---")
        for name, success, mae, exp_time in results:
            if success:
                print(f"  {name:<50} MAE: {mae:.4f}  ({exp_time/60:.1f} min)")
    
    if failed > 0:
        print("\n--- Failed Experiments ---")
        for name, success, mae, exp_time in results:
            if not success:
                print(f"  {name:<50} ({exp_time/60:.1f} min)")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run all experiment configs in a directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_group.py --config_path experiment_configs/group1_baselines
  python run_group.py --config_path /path/to/group2_sampling
        """
    )
    
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to directory containing YAML config files')
    
    args = parser.parse_args()
    
    # Setup logging with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    group_name = Path(args.config_path).name
    log_file = f'run_{group_name}_{timestamp}.log'
    
    setup_logging()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Logging to: {log_file}")
    
    # Run all configs
    run_all_configs_in_directory(args.config_path)
    
    logging.info(f"\nLog saved to: {log_file}")


if __name__ == '__main__':
    main()
