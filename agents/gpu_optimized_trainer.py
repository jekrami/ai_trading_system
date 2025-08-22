#!/usr/bin/env python3
"""
GPU-optimized training script for RTX 3090 with 24GB VRAM.
This script maximizes GPU utilization and training speed.
"""

import os
import sys
import json
import time
import logging
import argparse
import torch
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOTradingAgent
from agents.native_ppo_agent import NativePPOAgent, PPOBuffer
from envs.trading_env import TradingEnv
from agents.asset_universe import load_and_filter_data
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUOptimizedCallback(BaseCallback):
    """Callback optimized for GPU training with monitoring."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
                gpu_cached = torch.cuda.memory_reserved(0) / 1e9
                
                logger.info(f"Step {self.n_calls}: GPU Memory - "
                           f"Allocated: {gpu_allocated:.2f}GB/{gpu_memory:.2f}GB "
                           f"({gpu_allocated/gpu_memory*100:.1f}%), "
                           f"Cached: {gpu_cached:.2f}GB")
                
                # Log training speed
                elapsed = time.time() - self.start_time
                steps_per_sec = self.n_calls / elapsed
                logger.info(f"Training speed: {steps_per_sec:.2f} steps/sec")
        
        return True


def optimize_gpu_settings():
    """Optimize PyTorch settings for RTX 3090."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return "cpu"
    
    # Set optimal GPU settings for RTX 3090
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    # Enable TensorFloat-32 (TF32) for faster training on RTX 3090
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    device = "cuda:0"
    logger.info(f"GPU optimizations enabled for {torch.cuda.get_device_name(0)}")
    logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    
    return device


def create_optimized_agent(symbol: str, df, config: Dict, device: str) -> PPOTradingAgent:
    """Create a PPO agent optimized for RTX 3090."""
    
    # Extract training config
    training_config = config.get("training", {})
    
    # Create environment
    env = TradingEnv(
        df=df,
        window_size=training_config.get("window_size", 24),
        initial_balance=training_config.get("initial_balance", 10000.0),
        transaction_fee=training_config.get("transaction_fee", 0.001)
    )
    
    # Optimized policy kwargs for larger network
    policy_kwargs = {
        "net_arch": training_config.get("net_arch", [256, 256, 128]),
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True,
        "log_std_init": -2.0,
    }
    
    # Create agent with optimized parameters
    agent = PPOTradingAgent(
        symbol=symbol,
        env=env,
        device=device,
        n_steps=training_config.get("n_steps", 4096),
        batch_size=training_config.get("batch_size", 512),
        n_epochs=training_config.get("n_epochs", 20),
        learning_rate=training_config.get("learning_rate", 3e-4),
        policy_kwargs=policy_kwargs,
        use_mixed_precision=training_config.get("use_mixed_precision", True),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4)
    )
    
    return agent


def train_single_asset_native(symbol: str, data_path: str, config_path: str,
                             output_path: str, timesteps: int) -> Dict:
    """Train a single asset with native PyTorch PPO for maximum GPU utilization."""

    start_time = time.time()
    logger.info(f"Starting native PyTorch training for {symbol}")

    try:
        # Optimize GPU settings
        device = optimize_gpu_settings()

        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)

        training_config = config.get("training", {})

        # Load data
        asset_data = load_and_filter_data(data_path, symbols=[symbol])
        if symbol not in asset_data:
            raise ValueError(f"No data available for {symbol}")
        df = asset_data[symbol]

        logger.info(f"Loaded {len(df)} data points for {symbol}")

        # Create environment
        env = TradingEnv(
            df=df,
            window_size=training_config.get("window_size", 24),
            initial_balance=training_config.get("initial_balance", 10000.0),
            transaction_fee=training_config.get("transaction_fee", 0.001)
        )

        # Get observation and action dimensions
        obs = env.reset()
        obs_dim = len(obs)
        action_dim = env.action_space.n

        # Create native PPO agent
        agent = NativePPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=training_config.get("net_arch", [256, 256, 128]),
            lr=training_config.get("learning_rate", 3e-4),
            device=device,
            use_mixed_precision=training_config.get("use_mixed_precision", True)
        )

        # Training parameters
        n_steps = training_config.get("n_steps", 16384)
        batch_size = training_config.get("batch_size", 2048)
        n_epochs = training_config.get("n_epochs", 10)

        # Create buffer
        buffer = PPOBuffer(n_steps, obs_dim, device)

        # Training loop
        total_steps = 0
        episode_rewards = []

        logger.info(f"Training {symbol} for {timesteps} timesteps with native PyTorch")

        while total_steps < timesteps:
            # Collect experience
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(n_steps):
                if total_steps >= timesteps:
                    break

                # Get action
                action, log_prob, value = agent.get_action(obs)

                # Take step
                next_obs, reward, done, info = env.step(action)

                # Store experience
                buffer.store(obs, action, reward, value, log_prob, done)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if done:
                    episode_rewards.append(episode_reward)
                    obs = env.reset()
                    episode_reward = 0
                    episode_steps = 0

            # Update policy
            if buffer.size > 0:
                update_stats = agent.update(buffer, n_epochs, batch_size)
                buffer.reset()

                # Log progress
                if len(episode_rewards) > 0:
                    avg_reward = np.mean(episode_rewards[-10:])  # Last 10 episodes
                    logger.info(f"Steps: {total_steps}/{timesteps}, "
                               f"Avg Reward: {avg_reward:.4f}, "
                               f"Policy Loss: {update_stats['policy_loss']:.4f}")

        # Save model
        model_path = os.path.join(output_path, f"native_ppo_{symbol}.pt")
        agent.save(model_path)

        duration = time.time() - start_time
        final_stats = agent.get_stats()

        logger.info(f"Native training completed for {symbol} in {duration:.2f} seconds")
        logger.info(f"Final stats: {final_stats}")

        return {
            "success": True,
            "duration": duration,
            "timesteps": timesteps,
            "symbol": symbol,
            "final_stats": final_stats,
            "avg_episode_reward": np.mean(episode_rewards) if episode_rewards else 0
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Native training failed for {symbol}: {str(e)}")
        return {
            "success": False,
            "duration": duration,
            "error": str(e),
            "symbol": symbol
        }


def train_single_asset_optimized(symbol: str, data_path: str, config_path: str,
                                output_path: str, timesteps: int) -> Dict:
    """Train a single asset with GPU optimizations."""
    
    start_time = time.time()
    logger.info(f"Starting optimized training for {symbol}")
    
    try:
        # Optimize GPU settings
        device = optimize_gpu_settings()
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load data
        asset_data = load_and_filter_data(data_path, symbols=[symbol])
        if symbol not in asset_data:
            raise ValueError(f"No data available for {symbol}")
        df = asset_data[symbol]
        
        logger.info(f"Loaded {len(df)} data points for {symbol}")
        
        # Create optimized agent
        agent = create_optimized_agent(symbol, df, config, device)
        
        # Set up optimized callback
        callback = GPUOptimizedCallback(check_freq=1000)
        
        # Train with optimizations
        logger.info(f"Training {symbol} for {timesteps} timesteps with GPU optimizations")
        agent.train(total_timesteps=timesteps, callback=callback)
        
        # Save model
        model_path = os.path.join(output_path, f"ppo_{symbol}")
        agent.save(model_path)
        
        duration = time.time() - start_time
        logger.info(f"Training completed for {symbol} in {duration:.2f} seconds")
        
        return {
            "success": True,
            "duration": duration,
            "timesteps": timesteps,
            "symbol": symbol
        }
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Training failed for {symbol}: {str(e)}")
        return {
            "success": False,
            "duration": duration,
            "error": str(e),
            "symbol": symbol
        }


def train_multiple_assets_optimized(asset_list: List[str], data_path: str, 
                                   config_path: str, output_path: str, 
                                   timesteps: int, max_workers: int = 1) -> Dict:
    """Train multiple assets with optimized GPU utilization."""
    
    logger.info(f"Starting optimized training for {len(asset_list)} assets")
    logger.info(f"Using {max_workers} parallel workers")
    
    results = {}
    
    if max_workers == 1:
        # Sequential training for single GPU
        for symbol in asset_list:
            result = train_single_asset_optimized(
                symbol, data_path, config_path, output_path, timesteps
            )
            results[symbol] = result
    else:
        # Parallel training (if you have multiple GPUs or want CPU parallelism)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    train_single_asset_optimized, 
                    symbol, data_path, config_path, output_path, timesteps
                ): symbol 
                for symbol in asset_list
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Training failed for {symbol}: {str(e)}")
                    results[symbol] = {
                        "success": False,
                        "error": str(e),
                        "symbol": symbol
                    }
    
    # Print summary
    successful = sum(1 for r in results.values() if r.get("success", False))
    total_time = sum(r.get("duration", 0) for r in results.values())
    
    logger.info(f"Training complete: {successful}/{len(asset_list)} successful")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GPU-optimized training for RTX 3090")
    parser.add_argument("--symbol", type=str, help="Single asset symbol to train")
    parser.add_argument("--asset-list", type=str, help="Path to file with list of assets")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config-path", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--output-path", type=str, default="models", help="Output directory for models")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of training timesteps")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum parallel workers")
    
    args = parser.parse_args()
    
    # Determine asset list
    if args.symbol:
        asset_list = [args.symbol]
    elif args.asset_list:
        with open(args.asset_list, 'r') as f:
            asset_list = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must specify either --symbol or --asset-list")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run optimized training
    results = train_multiple_assets_optimized(
        asset_list=asset_list,
        data_path=args.data_path,
        config_path=args.config_path,
        output_path=args.output_path,
        timesteps=args.timesteps,
        max_workers=args.max_workers
    )
    
    # Save results
    results_path = os.path.join(args.output_path, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
