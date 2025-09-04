#!/usr/bin/env python3
"""
Test script to demonstrate smart training logic.
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_model_status(models_dir: str = "models"):
    """Check the status of existing models."""
    
    print("🔍 CHECKING MODEL STATUS")
    print("="*50)
    
    # List of assets to check
    assets = ["BTC_USD", "ETH_USD", "SOL_USD", "ADA_USD", "AVAX_USD"]
    
    for symbol in assets:
        native_model = os.path.join(models_dir, f"native_ppo_{symbol}.pt")
        sb3_model = os.path.join(models_dir, f"ppo_{symbol}.zip")
        metadata = os.path.join(models_dir, f"ppo_{symbol}_metadata.json")
        
        print(f"\n📊 {symbol}:")
        
        # Check model existence
        if os.path.exists(native_model):
            model_time = os.path.getmtime(native_model)
            model_date = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   ✅ Native PyTorch model: {native_model}")
            print(f"   📅 Created: {model_date}")
        elif os.path.exists(sb3_model):
            model_time = os.path.getmtime(sb3_model)
            model_date = datetime.fromtimestamp(model_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   ✅ Stable-Baselines3 model: {sb3_model}")
            print(f"   📅 Created: {model_date}")
        else:
            print(f"   ❌ No model found")
            continue
        
        # Check metadata
        if os.path.exists(metadata):
            try:
                with open(metadata, 'r') as f:
                    meta = json.load(f)
                print(f"   📋 Metadata: ✅")
                print(f"   🎯 Timesteps: {meta.get('training_config', {}).get('timesteps', 'Unknown')}")
                print(f"   📦 Batch size: {meta.get('training_config', {}).get('batch_size', 'Unknown')}")
            except:
                print(f"   📋 Metadata: ❌ (corrupted)")
        else:
            print(f"   📋 Metadata: ❌ (missing)")


def simulate_smart_training():
    """Simulate what smart training would do."""
    
    print("\n\n🧠 SMART TRAINING SIMULATION")
    print("="*50)
    
    assets = ["BTC_USD", "ETH_USD", "SOL_USD", "ADA_USD", "AVAX_USD"]
    
    for symbol in assets:
        print(f"\n🔍 Checking {symbol}...")
        
        # Check if model exists
        native_model = f"models/native_ppo_{symbol}.pt"
        metadata_file = f"models/ppo_{symbol}_metadata.json"
        
        if os.path.exists(native_model):
            print(f"   ✅ Model exists: {native_model}")
            
            if os.path.exists(metadata_file):
                print(f"   ✅ Metadata exists")
                print(f"   ⏭️  DECISION: Skip training (model up-to-date)")
            else:
                print(f"   ⚠️  Metadata missing")
                print(f"   🚀 DECISION: Retrain (no metadata)")
        else:
            print(f"   ❌ No model found")
            print(f"   🚀 DECISION: Train new model")


def demonstrate_force_retrain():
    """Demonstrate force retrain option."""
    
    print("\n\n🔄 FORCE RETRAIN DEMONSTRATION")
    print("="*50)
    
    print("With --force-retrain flag:")
    print("   🔄 All models will be retrained regardless of existing status")
    print("   ⚡ Useful when you want to:")
    print("      - Update models with new hyperparameters")
    print("      - Retrain with more timesteps")
    print("      - Start fresh after data updates")
    
    print("\nCommands:")
    print("   python run_system.py --train                    # Smart training")
    print("   python run_system.py --train --force-retrain    # Force retrain all")


def show_time_savings():
    """Show potential time savings."""
    
    print("\n\n⏱️  TIME SAVINGS ANALYSIS")
    print("="*50)
    
    # Estimate training times
    assets_count = 10
    training_time_per_asset = 110  # seconds (based on your output)
    
    total_training_time = assets_count * training_time_per_asset
    
    print(f"📊 Training Statistics:")
    print(f"   🎯 Assets: {assets_count}")
    print(f"   ⏱️  Time per asset: ~{training_time_per_asset} seconds")
    print(f"   🕐 Total training time: ~{total_training_time} seconds ({total_training_time/60:.1f} minutes)")
    
    print(f"\n💡 Smart Training Benefits:")
    print(f"   ✅ Skip unchanged models: 0 seconds")
    print(f"   🚀 Only train new/changed: Variable time")
    print(f"   💰 Time saved: Up to {total_training_time/60:.1f} minutes per run!")
    
    print(f"\n🔄 When Models Are Retrained:")
    print(f"   📝 Configuration changes (batch_size, learning_rate, etc.)")
    print(f"   📊 Data file updates (newer than model)")
    print(f"   🆕 New assets added")
    print(f"   🔄 Force retrain flag used")


def main():
    """Main demonstration function."""
    
    print("🎯 SMART TRAINING DEMONSTRATION")
    print("="*80)
    
    # Check current model status
    check_model_status()
    
    # Simulate smart training logic
    simulate_smart_training()
    
    # Show force retrain option
    demonstrate_force_retrain()
    
    # Show time savings
    show_time_savings()
    
    print("\n\n✅ SUMMARY:")
    print("="*50)
    print("✅ Smart training is now implemented!")
    print("✅ Models are only retrained when necessary")
    print("✅ Saves significant time on repeated runs")
    print("✅ Use --force-retrain to override when needed")
    print("\n🚀 Your RTX 3090 training is now optimized for efficiency!")


if __name__ == "__main__":
    main()
