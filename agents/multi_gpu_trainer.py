import os
import sys
import time
import json
import logging
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
import argparse
from datetime import datetime
import concurrent.futures
import multiprocessing

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        return list(range(gpu_count))
    return []

def allocate_gpus_to_assets(assets: List[str], gpu_ids: List[int]) -> Dict[str, int]:
    """Allocate GPUs to assets using round-robin allocation"""
    if not gpu_ids:
        return {asset: -1 for asset in assets}  # -1 indicates CPU
        
    allocation = {}
    for i, asset in enumerate(assets):
        gpu_idx = i % len(gpu_ids)
        allocation[asset] = gpu_ids[gpu_idx]
    
    return allocation

def train_agent_subprocess(
    symbol: str,
    gpu_id: int,
    data_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    timesteps: int = 100000,
    log_path: Optional[str] = None
) -> subprocess.Popen:
    """
    Launch a subprocess to train an agent on a specific GPU.
    
    Args:
        symbol: Asset symbol
        gpu_id: GPU ID (-1 for CPU)
        data_path: Path to the data directory
        output_path: Path to save the model
        config_path: Path to agent configuration file
        timesteps: Number of timesteps to train
        log_path: Path to save logs
        
    Returns:
        Subprocess object
    """
    # Create command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "train_single_agent.py"),
        "--symbol", symbol,
        "--data-path", data_path,
        "--output-path", output_path,
        "--timesteps", str(timesteps)
    ]
    
    if config_path:
        cmd.extend(["--config-path", config_path])
    
    if log_path:
        cmd.extend(["--log-path", log_path])
    
    # Set environment variables
    env = os.environ.copy()
    if gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    # Create log file
    if log_path is None:
        log_path = os.path.join(output_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, f"{symbol}_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Launch subprocess
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env
        )
    
    logger.info(f"Started training for {symbol} on GPU {gpu_id}, log: {log_file}")
    return process

def train_multiple_agents(
    asset_list: List[str],
    data_path: str,
    output_path: str,
    gpu_ids: Optional[List[int]] = None,
    config_path: Optional[str] = None,
    timesteps: int = 100000,
    max_parallel_jobs: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Train multiple agents in parallel.
    
    Args:
        asset_list: List of assets to train agents for
        data_path: Path to the data directory
        output_path: Path to save models
        gpu_ids: List of GPU IDs to use. If None, all available GPUs will be used.
        config_path: Path to agent configuration file
        timesteps: Number of timesteps to train
        max_parallel_jobs: Maximum number of jobs to run in parallel. If None, defaults to number of GPUs.
        
    Returns:
        Dictionary mapping asset symbols to training results
    """
    # Get available GPUs if not specified
    if gpu_ids is None:
        gpu_ids = get_available_gpus()
        if not gpu_ids:
            logger.warning("No GPUs detected, falling back to CPU")
            gpu_ids = [-1]  # CPU
    
    # Set max parallel jobs
    if max_parallel_jobs is None:
        if -1 in gpu_ids:  # CPU only
            max_parallel_jobs = multiprocessing.cpu_count() // 2
        else:
            max_parallel_jobs = len(gpu_ids)
    
    # Allocate GPUs to assets
    gpu_allocation = allocate_gpus_to_assets(asset_list, gpu_ids)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create log directory
    log_path = os.path.join(output_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    # Launch training processes in batches
    results = {}
    active_processes = {}
    completed_assets = []
    
    logger.info(f"Starting training for {len(asset_list)} assets on {len(gpu_ids)} GPUs")
    logger.info(f"GPU allocation: {gpu_allocation}")
    
    while len(completed_assets) < len(asset_list):
        # Check if we can launch more processes
        if len(active_processes) < max_parallel_jobs:
            # Get next asset that isn't being processed or completed
            available_assets = [a for a in asset_list 
                               if a not in active_processes.keys() 
                               and a not in completed_assets]
            
            if available_assets:
                symbol = available_assets[0]
                gpu_id = gpu_allocation[symbol]
                
                # Launch process
                process = train_agent_subprocess(
                    symbol=symbol,
                    gpu_id=gpu_id,
                    data_path=data_path,
                    output_path=output_path,
                    config_path=config_path,
                    timesteps=timesteps,
                    log_path=log_path
                )
                
                active_processes[symbol] = {
                    "process": process,
                    "start_time": time.time(),
                    "gpu_id": gpu_id
                }
        
        # Check for completed processes
        for symbol, proc_info in list(active_processes.items()):
            process = proc_info["process"]
            if process.poll() is not None:  # Process has finished
                # Record result
                exit_code = process.returncode
                end_time = time.time()
                duration = end_time - proc_info["start_time"]
                
                results[symbol] = {
                    "exit_code": exit_code,
                    "duration": duration,
                    "gpu_id": proc_info["gpu_id"],
                    "success": exit_code == 0
                }
                
                if exit_code == 0:
                    logger.info(f"Training for {symbol} completed successfully in {duration:.2f} seconds")
                else:
                    logger.error(f"Training for {symbol} failed with exit code {exit_code}")
                
                # Add to completed and remove from active
                completed_assets.append(symbol)
                del active_processes[symbol]
        
        # Sleep to prevent CPU hogging
        time.sleep(1)
    
    # Save results to JSON
    results_file = os.path.join(output_path, "training_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"All training jobs completed. Results saved to {results_file}")
    return results


def train_agents_with_threadpool(
    asset_list: List[str],
    data_path: str,
    output_path: str,
    gpu_ids: List[int],
    config_path: Optional[str] = None,
    timesteps: int = 100000,
    max_workers: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Alternative implementation using ThreadPoolExecutor.
    This approach works well when using external scripts.
    
    Args:
        asset_list: List of assets to train agents for
        data_path: Path to the data directory
        output_path: Path to save models
        gpu_ids: List of GPU IDs to use
        config_path: Path to agent configuration file
        timesteps: Number of timesteps to train
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Dictionary mapping asset symbols to training results
    """
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    # Allocate GPUs to assets
    gpu_allocation = allocate_gpus_to_assets(asset_list, gpu_ids)
    
    # Define worker function
    def train_worker(symbol):
        gpu_id = gpu_allocation[symbol]
        start_time = time.time()
        
        try:
            process = train_agent_subprocess(
                symbol=symbol,
                gpu_id=gpu_id,
                data_path=data_path,
                output_path=output_path,
                config_path=config_path,
                timesteps=timesteps,
                log_path=log_path
            )
            
            # Wait for process to complete
            exit_code = process.wait()
            duration = time.time() - start_time
            
            return symbol, {
                "exit_code": exit_code,
                "duration": duration,
                "gpu_id": gpu_id,
                "success": exit_code == 0
            }
        except Exception as e:
            logger.error(f"Error training {symbol}: {str(e)}")
            return symbol, {
                "exit_code": -1,
                "duration": time.time() - start_time,
                "gpu_id": gpu_id,
                "success": False,
                "error": str(e)
            }
    
    # Use ThreadPoolExecutor to manage workers
    results = {}
    max_workers = max_workers or len(gpu_ids)
    
    logger.info(f"Starting training pool with {max_workers} workers for {len(asset_list)} assets")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(train_worker, symbol): symbol for symbol in asset_list}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, result = future.result()
            results[symbol] = result
            
            if result["success"]:
                logger.info(f"Training for {symbol} completed successfully in {result['duration']:.2f} seconds")
            else:
                logger.error(f"Training for {symbol} failed with exit code {result['exit_code']}")
    
    # Save results to JSON
    results_file = os.path.join(output_path, "training_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"All training jobs completed. Results saved to {results_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Agent Training")
    parser.add_argument("--assets-file", type=str, required=True, help="JSON file with assets to train")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save models")
    parser.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--config-path", type=str, default=None, help="Path to agent configuration file")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to train")
    parser.add_argument("--max-parallel", type=int, default=None, help="Maximum number of parallel jobs")
    parser.add_argument("--use-threadpool", action="store_true", help="Use ThreadPool instead of manual process management")
    
    args = parser.parse_args()
    
    # Load assets from file
    with open(args.assets_file, "r") as f:
        assets_data = json.load(f)
        asset_list = assets_data.get("selected_assets", [])
    
    if not asset_list:
        logger.error(f"No assets found in {args.assets_file}")
        sys.exit(1)
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        try:
            gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        except ValueError:
            logger.error("Invalid GPU IDs specified. Use comma-separated integers.")
            sys.exit(1)
    
    # Train agents
    if args.use_threadpool:
        results = train_agents_with_threadpool(
            asset_list=asset_list,
            data_path=args.data_path,
            output_path=args.output_path,
            gpu_ids=gpu_ids or get_available_gpus() or [-1],
            config_path=args.config_path,
            timesteps=args.timesteps,
            max_workers=args.max_parallel
        )
    else:
        results = train_multiple_agents(
            asset_list=asset_list,
            data_path=args.data_path,
            output_path=args.output_path,
            gpu_ids=gpu_ids,
            config_path=args.config_path,
            timesteps=args.timesteps,
            max_parallel_jobs=args.max_parallel
        )
    
    # Print summary
    success_count = sum(1 for r in results.values() if r.get("success", False))
    logger.info(f"Training complete: {success_count}/{len(asset_list)} agents trained successfully") 