#!/usr/bin/env python3
"""
Quick training script optimized for RTX 3090.
This script automatically uses the best settings for your GPU.
"""

import os
import sys
import json
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gpu_optimized_trainer import train_single_asset_optimized, train_single_asset_native, train_multiple_assets_optimized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_rtx3090_config(base_config_path: str = "config.json") -> str:
    """Create an optimized config file for RTX 3090."""
    
    # Load base config
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # RTX 3090 optimized settings
    rtx3090_training_config = {
        "window_size": 24,
        "initial_balance": 10000.0,
        "transaction_fee": 0.001,
        "timesteps": 100000,
        "batch_size": 512,           # Optimized for 24GB VRAM
        "n_steps": 4096,             # 8x batch size
        "n_epochs": 20,              # More epochs for better learning
        "learning_rate": 0.0003,
        "net_arch": [256, 256, 128], # Larger network
        "use_mixed_precision": True,  # Faster training on RTX 3090
        "gradient_accumulation_steps": 4  # Better gradient estimates
    }
    
    # Update config
    config["training"] = rtx3090_training_config
    
    # Save optimized config
    optimized_config_path = "config_rtx3090.json"
    with open(optimized_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"RTX 3090 optimized config saved to {optimized_config_path}")
    return optimized_config_path


def main():
    parser = argparse.ArgumentParser(description="Quick training for RTX 3090")
    parser.add_argument("--symbol", type=str, help="Single asset symbol (e.g., BTC_USD)")
    parser.add_argument("--symbols", type=str, nargs="+", help="Multiple symbols")
    parser.add_argument("--data-path", type=str, default="data", help="Data directory")
    parser.add_argument("--output-path", type=str, default="models", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--config", type=str, default="config.json", help="Base config file")
    parser.add_argument("--native", action="store_true", help="Use native PyTorch implementation for maximum GPU utilization")
    
    args = parser.parse_args()
    
    # Create optimized config
    optimized_config = create_rtx3090_config(args.config)
    
    # Determine symbols to train
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    else:
        # Default symbols
        symbols = ["BTC_USD", "ETH_USD", "SOL_USD"]
        logger.info(f"No symbols specified, using default: {symbols}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    logger.info("="*60)
    logger.info("RTX 3090 OPTIMIZED TRAINING")
    logger.info("="*60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Config: {optimized_config}")
    logger.info("="*60)
    
    # Train assets
    if len(symbols) == 1:
        # Single asset training
        if args.native:
            logger.info("Using native PyTorch implementation for maximum GPU utilization")
            result = train_single_asset_native(
                symbol=symbols[0],
                data_path=args.data_path,
                config_path=optimized_config,
                output_path=args.output_path,
                timesteps=args.timesteps
            )
        else:
            result = train_single_asset_optimized(
                symbol=symbols[0],
                data_path=args.data_path,
                config_path=optimized_config,
                output_path=args.output_path,
                timesteps=args.timesteps
            )
        
        if result["success"]:
            logger.info(f"✅ Training completed successfully in {result['duration']:.2f} seconds")
        else:
            logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    else:
        # Multiple asset training
        results = train_multiple_assets_optimized(
            asset_list=symbols,
            data_path=args.data_path,
            config_path=optimized_config,
            output_path=args.output_path,
            timesteps=args.timesteps,
            max_workers=1  # Sequential for single GPU
        )
        
        # Print summary
        successful = sum(1 for r in results.values() if r.get("success", False))
        total_time = sum(r.get("duration", 0) for r in results.values())
        
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Successful: {successful}/{len(symbols)}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per asset: {total_time/len(symbols):.2f} seconds")
        
        for symbol, result in results.items():
            status = "✅" if result.get("success", False) else "❌"
            duration = result.get("duration", 0)
            logger.info(f"{status} {symbol}: {duration:.2f}s")
    
    logger.info("Training complete! 🚀")


if __name__ == "__main__":
    main()
