#!/usr/bin/env python3
"""
GPU benchmark and monitoring script for RTX 3090 optimization.
"""

import os
import sys
import time
import torch
import psutil
import logging
import subprocess
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU utilization and performance."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device_count = torch.cuda.device_count()
            self.device_name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
        
    def get_gpu_info(self) -> Dict:
        """Get current GPU information."""
        if not self.gpu_available:
            return {"available": False}
        
        info = {
            "available": True,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "total_memory_gb": self.total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
            "cached_memory_gb": torch.cuda.memory_reserved(0) / 1e9,
            "free_memory_gb": (self.total_memory - torch.cuda.memory_reserved(0)) / 1e9
        }
        
        # Calculate utilization percentages
        info["memory_utilization"] = (info["allocated_memory_gb"] / info["total_memory_gb"]) * 100
        info["cache_utilization"] = (info["cached_memory_gb"] / info["total_memory_gb"]) * 100
        
        return info
    
    def benchmark_batch_sizes(self, model_size: str = "medium") -> Dict:
        """Benchmark different batch sizes to find optimal configuration."""
        if not self.gpu_available:
            logger.warning("GPU not available for benchmarking")
            return {}
        
        logger.info("Starting batch size benchmark...")
        
        # Define model architectures
        architectures = {
            "small": [64, 64],
            "medium": [256, 256, 128],
            "large": [512, 512, 256, 128]
        }
        
        net_arch = architectures.get(model_size, architectures["medium"])
        input_size = 266  # Based on your trading environment
        
        # Test different batch sizes
        batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Create a simple model for testing
                model = torch.nn.Sequential(
                    torch.nn.Linear(input_size, net_arch[0]),
                    torch.nn.ReLU(),
                    *[torch.nn.Sequential(
                        torch.nn.Linear(net_arch[i], net_arch[i+1]),
                        torch.nn.ReLU()
                    ) for i in range(len(net_arch)-1)],
                    torch.nn.Linear(net_arch[-1], 3)  # 3 actions
                ).cuda()
                
                # Create dummy data
                dummy_input = torch.randn(batch_size, input_size).cuda()
                dummy_target = torch.randint(0, 3, (batch_size,)).cuda()
                
                # Measure memory and time
                start_memory = torch.cuda.memory_allocated(0)
                start_time = time.time()
                
                # Forward and backward pass
                optimizer = torch.optim.Adam(model.parameters())
                for _ in range(10):  # Multiple iterations for better measurement
                    optimizer.zero_grad()
                    output = model(dummy_input)
                    loss = torch.nn.CrossEntropyLoss()(output, dummy_target)
                    loss.backward()
                    optimizer.step()
                
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                
                end_time = time.time()
                peak_memory = torch.cuda.max_memory_allocated(0)
                
                # Calculate metrics
                time_per_batch = (end_time - start_time) / 10
                memory_used = (peak_memory - start_memory) / 1e9
                throughput = batch_size / time_per_batch
                
                results[batch_size] = {
                    "time_per_batch": time_per_batch,
                    "memory_used_gb": memory_used,
                    "throughput": throughput,
                    "success": True
                }
                
                logger.info(f"Batch size {batch_size}: {time_per_batch:.4f}s/batch, "
                           f"{memory_used:.2f}GB memory, {throughput:.1f} samples/s")
                
                # Clean up
                del model, dummy_input, dummy_target, optimizer
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"Batch size {batch_size}: Out of memory")
                    results[batch_size] = {"success": False, "error": "OOM"}
                    break
                else:
                    logger.error(f"Batch size {batch_size}: {str(e)}")
                    results[batch_size] = {"success": False, "error": str(e)}
        
        return results
    
    def recommend_settings(self, benchmark_results: Dict) -> Dict:
        """Recommend optimal settings based on benchmark results."""
        if not benchmark_results:
            return {}
        
        # Find the largest successful batch size
        successful_batches = {k: v for k, v in benchmark_results.items() 
                             if v.get("success", False)}
        
        if not successful_batches:
            return {"error": "No successful batch sizes found"}
        
        # Find optimal batch size (highest throughput that uses <80% memory)
        gpu_info = self.get_gpu_info()
        max_memory_gb = gpu_info["total_memory_gb"] * 0.8  # Use 80% of available memory
        
        optimal_batch = None
        max_throughput = 0
        
        for batch_size, results in successful_batches.items():
            if results["memory_used_gb"] <= max_memory_gb:
                if results["throughput"] > max_throughput:
                    max_throughput = results["throughput"]
                    optimal_batch = batch_size
        
        if optimal_batch is None:
            # If no batch fits in 80% memory, use the largest successful one
            optimal_batch = max(successful_batches.keys())
        
        recommendations = {
            "optimal_batch_size": optimal_batch,
            "max_safe_batch_size": max(successful_batches.keys()),
            "recommended_n_steps": optimal_batch * 8,  # 8x batch size for n_steps
            "recommended_n_epochs": min(20, max(10, 2048 // optimal_batch)),
            "memory_utilization": successful_batches[optimal_batch]["memory_used_gb"] / gpu_info["total_memory_gb"] * 100
        }
        
        return recommendations


def run_gpu_benchmark():
    """Run comprehensive GPU benchmark."""
    monitor = GPUMonitor()
    
    # Print GPU information
    gpu_info = monitor.get_gpu_info()
    print("\n" + "="*50)
    print("GPU INFORMATION")
    print("="*50)
    
    if gpu_info["available"]:
        print(f"Device: {gpu_info['device_name']}")
        print(f"Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"Free Memory: {gpu_info['free_memory_gb']:.2f} GB")
    else:
        print("No GPU available")
        return
    
    # Run benchmark
    print("\n" + "="*50)
    print("BATCH SIZE BENCHMARK")
    print("="*50)
    
    benchmark_results = monitor.benchmark_batch_sizes("medium")
    
    # Get recommendations
    recommendations = monitor.recommend_settings(benchmark_results)
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS FOR RTX 3090")
    print("="*50)
    
    if "error" not in recommendations:
        print(f"Optimal Batch Size: {recommendations['optimal_batch_size']}")
        print(f"Recommended n_steps: {recommendations['recommended_n_steps']}")
        print(f"Recommended n_epochs: {recommendations['recommended_n_epochs']}")
        print(f"Memory Utilization: {recommendations['memory_utilization']:.1f}%")
        
        # Generate optimized config
        optimized_config = {
            "training": {
                "batch_size": recommendations['optimal_batch_size'],
                "n_steps": recommendations['recommended_n_steps'],
                "n_epochs": recommendations['recommended_n_epochs'],
                "net_arch": [256, 256, 128],
                "use_mixed_precision": True,
                "gradient_accumulation_steps": 4
            }
        }
        
        print("\nOptimized config snippet:")
        print(optimized_config)
        
    else:
        print(f"Error: {recommendations['error']}")
    
    return benchmark_results, recommendations


def monitor_training_performance(duration_minutes: int = 5):
    """Monitor GPU performance during training."""
    monitor = GPUMonitor()
    
    if not monitor.gpu_available:
        logger.warning("GPU not available for monitoring")
        return
    
    logger.info(f"Monitoring GPU performance for {duration_minutes} minutes...")
    
    timestamps = []
    memory_usage = []
    memory_cached = []
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            gpu_info = monitor.get_gpu_info()
            current_time = time.time() - start_time
            
            timestamps.append(current_time)
            memory_usage.append(gpu_info["memory_utilization"])
            memory_cached.append(gpu_info["cache_utilization"])
            
            logger.info(f"Time: {current_time:.1f}s, "
                       f"Memory: {gpu_info['memory_utilization']:.1f}%, "
                       f"Cached: {gpu_info['cache_utilization']:.1f}%")
            
            time.sleep(10)  # Monitor every 10 seconds
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, memory_usage, label="Memory Usage", linewidth=2)
    plt.plot(timestamps, memory_cached, label="Cached Memory", linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("GPU Memory Utilization (%)")
    plt.title("GPU Memory Utilization During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = "gpu_monitoring.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Monitoring plot saved to {plot_path}")
    
    return timestamps, memory_usage, memory_cached


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU benchmark and monitoring")
    parser.add_argument("--benchmark", action="store_true", help="Run GPU benchmark")
    parser.add_argument("--monitor", type=int, help="Monitor GPU for N minutes")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_gpu_benchmark()
    elif args.monitor:
        monitor_training_performance(args.monitor)
    else:
        # Run benchmark by default
        run_gpu_benchmark()
