#!/usr/bin/env python3
"""
Configuration validation script for config.json.
Helps users check their configuration and provides recommendations.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta

def validate_system_config(system_config):
    """Validate system configuration section."""
    issues = []
    warnings = []
    
    # Check required fields
    required_fields = ["name", "version", "mode", "start_date", "end_date"]
    for field in required_fields:
        if field not in system_config:
            issues.append(f"Missing required field: system.{field}")
    
    # Validate mode
    valid_modes = ["backtest", "live", "paper"]
    if "mode" in system_config and system_config["mode"] not in valid_modes:
        issues.append(f"Invalid mode '{system_config['mode']}'. Must be one of: {valid_modes}")
    
    # Validate dates
    try:
        if "start_date" in system_config and "end_date" in system_config:
            start = datetime.strptime(system_config["start_date"], "%Y-%m-%d")
            end = datetime.strptime(system_config["end_date"], "%Y-%m-%d")
            
            if start >= end:
                issues.append("start_date must be before end_date")
            
            # Check if date range is reasonable
            days_diff = (end - start).days
            if days_diff < 30:
                warnings.append(f"Date range is only {days_diff} days. Consider at least 90 days for better training.")
            elif days_diff > 730:
                warnings.append(f"Date range is {days_diff} days. Very long periods may slow training.")
    except ValueError as e:
        issues.append(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")
    
    # Warn about live mode
    if system_config.get("mode") == "live":
        warnings.append("⚠️  LIVE MODE DETECTED! Make sure you've thoroughly tested in backtest mode first.")
    
    return issues, warnings


def validate_asset_selection(asset_config):
    """Validate asset selection configuration."""
    issues = []
    warnings = []
    
    # Check max_assets
    max_assets = asset_config.get("max_assets", 5)
    if max_assets < 1:
        issues.append("max_assets must be at least 1")
    elif max_assets > 20:
        warnings.append(f"max_assets is {max_assets}. Consider 3-10 for better performance.")
    elif max_assets > 10:
        warnings.append(f"max_assets is {max_assets}. This may slow training significantly.")
    
    # Check volume requirements
    min_volume = asset_config.get("min_volume_usd", 1000000)
    if min_volume < 100000:
        warnings.append(f"min_volume_usd is ${min_volume:,}. Consider at least $1M for liquidity.")
    
    # Check market cap
    min_market_cap = asset_config.get("min_market_cap_usd", 10000000)
    if min_market_cap < 1000000:
        warnings.append(f"min_market_cap_usd is ${min_market_cap:,}. Very small caps can be risky.")
    
    # Check correlation threshold
    correlation = asset_config.get("correlation_threshold", 0.8)
    if correlation < 0.3 or correlation > 1.0:
        issues.append("correlation_threshold must be between 0.3 and 1.0")
    elif correlation < 0.5:
        warnings.append("Very low correlation_threshold may limit asset selection.")
    
    return issues, warnings


def validate_training_config(training_config):
    """Validate training configuration."""
    issues = []
    warnings = []
    
    # Check timesteps
    timesteps = training_config.get("timesteps", 100000)
    if timesteps < 10000:
        warnings.append(f"timesteps is {timesteps:,}. Consider at least 50,000 for decent training.")
    elif timesteps > 1000000:
        warnings.append(f"timesteps is {timesteps:,}. This will take a very long time to train.")
    
    # Check batch size
    batch_size = training_config.get("batch_size", 64)
    if batch_size < 32:
        warnings.append("batch_size is very small. Consider at least 64.")
    elif batch_size > 4096:
        warnings.append("batch_size is very large. May cause out-of-memory errors.")
    
    # Check if RTX 3090 optimized
    if batch_size >= 2048 and training_config.get("use_native_pytorch", False):
        print("✅ RTX 3090 optimizations detected!")
    elif batch_size < 1024:
        warnings.append("Consider RTX 3090 optimizations: batch_size=2048, use_native_pytorch=true")
    
    # Check learning rate
    lr = training_config.get("learning_rate", 0.0003)
    if lr > 0.01:
        warnings.append("learning_rate is very high. May cause unstable training.")
    elif lr < 0.00001:
        warnings.append("learning_rate is very low. Training may be very slow.")
    
    # Check network architecture
    net_arch = training_config.get("net_arch", [64, 64])
    if len(net_arch) < 2:
        warnings.append("net_arch should have at least 2 layers.")
    elif max(net_arch) > 1024:
        warnings.append("Very large network layers may cause memory issues.")
    
    # Check balance
    balance = training_config.get("initial_balance", 10000)
    if balance < 1000:
        warnings.append("initial_balance is very low. Consider at least $10,000 for realistic testing.")
    
    return issues, warnings


def validate_portfolio_config(portfolio_config):
    """Validate portfolio configuration."""
    issues = []
    warnings = []
    
    # Check allocation method
    valid_methods = ["equal", "risk_parity", "markowitz"]
    method = portfolio_config.get("allocation_method", "risk_parity")
    if method not in valid_methods:
        issues.append(f"Invalid allocation_method '{method}'. Must be one of: {valid_methods}")
    
    # Check allocation limits
    max_alloc = portfolio_config.get("max_allocation", 0.4)
    min_alloc = portfolio_config.get("min_allocation", 0.05)
    
    if max_alloc <= min_alloc:
        issues.append("max_allocation must be greater than min_allocation")
    
    if max_alloc > 0.8:
        warnings.append("max_allocation > 80% is very risky. Consider 40% or less.")
    
    if min_alloc < 0.01:
        warnings.append("min_allocation < 1% may be too small to be meaningful.")
    
    # Check stop loss and take profit
    stop_loss = portfolio_config.get("stop_loss")
    take_profit = portfolio_config.get("take_profit")
    
    if stop_loss and (stop_loss < 0.02 or stop_loss > 0.5):
        warnings.append("stop_loss should typically be between 2% and 50%.")
    
    if take_profit and (take_profit < 0.05 or take_profit > 1.0):
        warnings.append("take_profit should typically be between 5% and 100%.")
    
    if stop_loss and take_profit and stop_loss >= take_profit:
        warnings.append("stop_loss should be less than take_profit.")
    
    return issues, warnings


def validate_config_file(config_path):
    """Validate entire configuration file."""
    print(f"🔍 Validating configuration: {config_path}")
    print("="*60)
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {str(e)}")
        return False
    
    all_issues = []
    all_warnings = []
    
    # Validate each section
    sections = [
        ("system", validate_system_config),
        ("asset_selection", validate_asset_selection),
        ("training", validate_training_config),
        ("portfolio", validate_portfolio_config)
    ]
    
    for section_name, validator in sections:
        if section_name in config:
            issues, warnings = validator(config[section_name])
            all_issues.extend([f"{section_name}.{issue}" for issue in issues])
            all_warnings.extend([f"{section_name}.{warning}" for warning in warnings])
        else:
            all_warnings.append(f"Missing optional section: {section_name}")
    
    # Print results
    if all_issues:
        print("❌ CRITICAL ISSUES:")
        for issue in all_issues:
            print(f"   • {issue}")
        print()
    
    if all_warnings:
        print("⚠️  WARNINGS:")
        for warning in all_warnings:
            print(f"   • {warning}")
        print()
    
    if not all_issues and not all_warnings:
        print("✅ Configuration looks good!")
    elif not all_issues:
        print("✅ Configuration is valid (with warnings above)")
    else:
        print("❌ Configuration has critical issues that must be fixed")
    
    # Provide recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    # Beginner recommendations
    if config.get("training", {}).get("timesteps", 100000) > 200000:
        print("   • Consider reducing timesteps to 100,000 for faster testing")
    
    if config.get("asset_selection", {}).get("max_assets", 5) > 5:
        print("   • Consider starting with 3-5 assets for simplicity")
    
    if config.get("system", {}).get("mode") != "backtest":
        print("   • Use 'backtest' mode for testing before going live")
    
    print("   • See CONFIG_GUIDE.md for detailed explanations")
    print("   • Start with beginner settings and gradually advance")
    
    return len(all_issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate AI Trading System configuration")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    
    args = parser.parse_args()
    
    print("🎯 AI Trading System Configuration Validator")
    print("="*60)
    
    is_valid = validate_config_file(args.config)
    
    if is_valid:
        print(f"\n🚀 Ready to run: python run_system.py --pipeline")
    else:
        print(f"\n🔧 Please fix the issues above before running the system")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())
