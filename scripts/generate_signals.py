#!/usr/bin/env python3
"""
Simple Trading Signal Generator - Shows what to buy/sell/hold and when.
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.asset_universe import load_and_filter_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_latest_prices(data_dir: str = "data") -> dict:
    """Get the latest prices for all available assets."""
    logger.info("Loading latest market prices...")
    
    # Load top assets
    try:
        with open("outputs/top_assets.json", 'r') as f:
            top_assets = json.load(f)
            symbols = top_assets["selected_assets"]
    except:
        symbols = ["BTC_USD", "ETH_USD", "SOL_USD", "LINK_USD", "XRP_USD"]
    
    prices = {}
    asset_data = load_and_filter_data(data_dir, symbols=symbols)
    
    for symbol, df in asset_data.items():
        latest_data = df.iloc[-1]
        prices[symbol] = {
            "current_price": float(latest_data['close']),
            "previous_price": float(df.iloc[-2]['close']),
            "volume": float(latest_data['volume']),
            "high_24h": float(df.iloc[-24:]['high'].max()),
            "low_24h": float(df.iloc[-24:]['low'].min()),
            "change_24h": float((latest_data['close'] - df.iloc[-25]['close']) / df.iloc[-25]['close'] * 100),
            "timestamp": str(df.index[-1])
        }
    
    return prices


def generate_simple_signals(prices: dict, portfolio_balance: float = 10000.0) -> dict:
    """Generate simple trading signals based on price movements and trends."""
    logger.info("Generating AI trading signals...")
    
    signals = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_balance": portfolio_balance,
        "signals": {},
        "summary": {
            "total_buy_signals": 0,
            "total_sell_signals": 0,
            "total_hold_signals": 0,
            "total_trade_value": 0.0
        }
    }
    
    # Simple trading logic (replace with your trained models)
    for symbol, data in prices.items():
        current_price = data["current_price"]
        change_24h = data["change_24h"]
        
        # Simple momentum-based signals
        if change_24h > 2.0:  # Strong upward momentum
            action = "BUY"
            confidence = min(0.8, abs(change_24h) / 10.0)
            position_size = (portfolio_balance * 0.1) / current_price  # 10% of portfolio
        elif change_24h < -2.0:  # Strong downward momentum
            action = "SELL"
            confidence = min(0.8, abs(change_24h) / 10.0)
            position_size = (portfolio_balance * 0.05) / current_price  # 5% of portfolio
        else:  # Sideways movement
            action = "HOLD"
            confidence = 0.6
            position_size = 0.0
        
        trade_value = position_size * current_price
        
        signal = {
            "symbol": symbol,
            "action": action,
            "confidence": round(confidence, 3),
            "current_price": round(current_price, 4),
            "change_24h": round(change_24h, 2),
            "position_size": round(position_size, 6),
            "trade_value_usd": round(trade_value, 2),
            "timestamp": datetime.now().isoformat(),
            "reasoning": get_reasoning(action, change_24h, current_price)
        }
        
        signals["signals"][symbol] = signal
        
        # Update summary
        if action == "BUY":
            signals["summary"]["total_buy_signals"] += 1
        elif action == "SELL":
            signals["summary"]["total_sell_signals"] += 1
        else:
            signals["summary"]["total_hold_signals"] += 1
        
        signals["summary"]["total_trade_value"] += trade_value
    
    return signals


def get_reasoning(action: str, change_24h: float, price: float) -> str:
    """Generate reasoning for the trading signal."""
    if action == "BUY":
        return f"Strong upward momentum (+{change_24h:.1f}% in 24h) suggests buying opportunity at ${price:.4f}"
    elif action == "SELL":
        return f"Strong downward momentum ({change_24h:.1f}% in 24h) suggests selling at ${price:.4f}"
    else:
        return f"Sideways movement ({change_24h:.1f}% in 24h) suggests holding position at ${price:.4f}"


def print_trading_signals(signals: dict):
    """Print formatted trading signals."""
    print("\n" + "="*80)
    print("🚀 AI TRADING SIGNALS - WHAT TO BUY/SELL/HOLD")
    print("="*80)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Portfolio Balance: ${signals['portfolio_balance']:,.2f}")
    print(f"📊 Total Assets: {len(signals['signals'])}")
    print()
    
    # Sort by action priority (BUY first, then SELL, then HOLD)
    action_priority = {"BUY": 1, "SELL": 2, "HOLD": 3}
    sorted_signals = sorted(
        signals["signals"].items(), 
        key=lambda x: action_priority[x[1]["action"]]
    )
    
    for symbol, signal in sorted_signals:
        action_emoji = {"BUY": "🟢 BUY ", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}[signal["action"]]
        
        print(f"{action_emoji} {signal['symbol']}")
        print(f"   💵 Current Price: ${signal['current_price']:,.4f}")
        print(f"   📈 24h Change: {signal['change_24h']:+.2f}%")
        
        if signal["action"] != "HOLD":
            print(f"   📏 Recommended Size: {signal['position_size']:.6f} units")
            print(f"   💲 Trade Value: ${signal['trade_value_usd']:,.2f}")
        
        print(f"   🎯 Confidence: {signal['confidence']:.1%}")
        print(f"   💭 Reasoning: {signal['reasoning']}")
        print()
    
    summary = signals["summary"]
    print("📈 TRADING SUMMARY:")
    print(f"   🟢 BUY Signals: {summary['total_buy_signals']}")
    print(f"   🔴 SELL Signals: {summary['total_sell_signals']}")
    print(f"   🟡 HOLD Signals: {summary['total_hold_signals']}")
    print(f"   💰 Total Trade Value: ${summary['total_trade_value']:,.2f}")
    print()
    print("⚠️  NOTE: These are AI-generated signals for educational purposes.")
    print("   Always do your own research before making trading decisions!")
    print("="*80)


def save_signals(signals: dict, output_path: str = "outputs/trading_signals.json"):
    """Save signals to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(signals, f, indent=2)
    
    logger.info(f"Trading signals saved to {output_path}")


def main():
    """Generate and display trading signals."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Trading Signals")
    parser.add_argument("--balance", type=float, default=10000.0, help="Portfolio balance")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output", type=str, default="outputs/trading_signals.json", help="Output file")
    
    args = parser.parse_args()
    
    try:
        # Get latest prices
        prices = get_latest_prices(args.data_dir)
        
        # Generate signals
        signals = generate_simple_signals(prices, args.balance)
        
        # Display results
        print_trading_signals(signals)
        
        # Save to file
        save_signals(signals, args.output)
        
        print(f"\n✅ Trading signals generated successfully!")
        print(f"📁 Saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
