#!/usr/bin/env python
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Import system components
from agents.asset_universe import AssetUniverseSelector, load_and_filter_data
from agents.ppo_agent import PPOTradingAgent
from agents.multi_gpu_trainer import train_multiple_agents, get_available_gpus
from agents.gpu_optimized_trainer import train_single_asset_native, train_single_asset_optimized
from envs.trading_env import TradingEnv
from meta_agent.portfolio_allocator import PortfolioAllocator
from backtesting.backtest_engine import BacktestEngine, sma_crossover_strategy
from monitoring.metrics_logger import MetricsLogger
from llm.reasoning_engine import ReasoningEngine
from trading.signal_generator import TradingSignalGenerator

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystem:
    """Main class for orchestrating the AI trading system"""
    
    def __init__(
        self,
        config_path: str = "config.json",
        data_dir: str = "data",
        output_dir: str = "outputs",
        models_dir: str = "models",
        logs_dir: str = "logs"
    ):
        """
        Initialize the trading system.
        
        Args:
            config_path: Path to configuration file
            data_dir: Directory for market data
            output_dir: Directory for output files
            models_dir: Directory for model files
            logs_dir: Directory for logs
        """
        self.config_path = config_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        
        # Create directories if they don't exist
        for d in [data_dir, output_dir, models_dir, logs_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.metrics_logger = MetricsLogger(logs_dir)
        
        # Runtime variables
        self.selected_assets = []
        self.asset_data = {}
        self.trained_agents = {}
        self.portfolio_allocator = None
        self.force_retrain = False  # Flag to force retraining
    
    def _load_config(self) -> dict:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                return self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self._create_default_config()
    
    def _create_default_config(self) -> dict:
        """Create default configuration"""
        config = {
            "system": {
                "name": "AI Trading System",
                "version": "1.0.0",
                "mode": "backtest",  # backtest, paper, live
                "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d")
            },
            "asset_selection": {
                "max_assets": 5,
                "lookback_days": 30,
                "min_volume_usd": 1000000,
                "min_market_cap_usd": 10000000
            },
            "training": {
                "window_size": 24,
                "initial_balance": 10000.0,
                "transaction_fee": 0.001,
                "timesteps": 100000,
                "batch_size": 64,
                "learning_rate": 3e-4
            },
            "portfolio": {
                "allocation_method": "risk_parity",  # risk_parity, markowitz, equal, kelly
                "max_allocation": 0.4,
                "min_allocation": 0.05,
                "risk_aversion": 1.2
            },
            "execution": {
                "rebalance_interval": "1d",  # 1d, 1h, etc.
                "order_type": "market"
            }
        }
        
        # Save default config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Default configuration created at {self.config_path}")
        return config
    
    def select_assets(self) -> list:
        """Select assets to trade"""
        logger.info("Starting asset selection")
        
        # Get configuration
        asset_config = self.config.get("asset_selection", {})
        
        # Initialize asset selector
        selector = AssetUniverseSelector(
            data_dir=self.data_dir,
            lookback_days=asset_config.get("lookback_days", 30),
            max_assets=asset_config.get("max_assets", 5),
            min_volume_usd=asset_config.get("min_volume_usd", 1000000),
            min_market_cap_usd=asset_config.get("min_market_cap_usd", 10000000)
        )
        
        # Select assets
        selection_results = selector.select_assets()
        self.selected_assets = selection_results.get("selected_assets", [])
        
        # Log results
        if self.selected_assets:
            logger.info(f"Selected {len(self.selected_assets)} assets: {', '.join(self.selected_assets)}")
        else:
            logger.warning("No assets selected!")
        
        return self.selected_assets
    
    def load_data(self) -> dict:
        """Load market data for selected assets"""
        logger.info("Loading market data")
        
        # Get configuration
        system_config = self.config.get("system", {})
        start_date = system_config.get("start_date", None)
        end_date = system_config.get("end_date", None)
        
        # Load data
        assets_to_load = self.selected_assets if self.selected_assets else None
        self.asset_data = load_and_filter_data(
            data_dir=self.data_dir,
            symbols=assets_to_load,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Loaded data for {len(self.asset_data)} assets")
        return self.asset_data
    
    def _should_retrain_agent(self, symbol: str, training_config: Dict) -> bool:
        """Check if an agent needs retraining based on existing models and data changes."""

        # Check for existing models
        native_model_path = os.path.join(self.models_dir, f"native_ppo_{symbol}.pt")
        sb3_model_path = os.path.join(self.models_dir, f"ppo_{symbol}.zip")
        metadata_path = os.path.join(self.models_dir, f"ppo_{symbol}_metadata.json")

        model_exists = os.path.exists(native_model_path) or os.path.exists(sb3_model_path)

        if not model_exists:
            logger.info(f"No existing model found for {symbol}, training required")
            return True

        # Check force retrain flag
        if getattr(self, 'force_retrain', False):
            logger.info(f"Force retrain enabled for {symbol}")
            return True

        # Check if metadata exists and is recent
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Check if training config has changed
                stored_config = metadata.get("training_config", {})
                current_key_config = {
                    "timesteps": training_config.get("timesteps", 100000),
                    "batch_size": training_config.get("batch_size", 64),
                    "learning_rate": training_config.get("learning_rate", 3e-4),
                    "net_arch": training_config.get("net_arch", [64, 64])
                }

                if stored_config != current_key_config:
                    logger.info(f"Training configuration changed for {symbol}, retraining required")
                    return True

                # Check data freshness (if data file is newer than model)
                data_files = [
                    os.path.join(self.data_dir, f"{symbol.lower()}_1h.csv"),
                    os.path.join(self.data_dir, '1h', f"{symbol.upper()}_1h.csv"),
                    os.path.join(self.data_dir, f"{symbol.upper()}.csv")
                ]

                model_time = os.path.getmtime(native_model_path if os.path.exists(native_model_path) else sb3_model_path)

                for data_file in data_files:
                    if os.path.exists(data_file):
                        data_time = os.path.getmtime(data_file)
                        if data_time > model_time:
                            logger.info(f"Data file {data_file} is newer than model for {symbol}, retraining required")
                            return True
                        break

                logger.info(f"Existing model for {symbol} is up-to-date, skipping training")
                return False

            except Exception as e:
                logger.warning(f"Error reading metadata for {symbol}: {str(e)}, will retrain")
                return True

        # If no metadata, assume we need to retrain
        logger.info(f"No metadata found for {symbol}, training required")
        return True

    def _save_training_metadata(self, symbol: str, training_config: Dict, model_path: str):
        """Save training metadata for future reference."""
        metadata = {
            "symbol": symbol,
            "train_date": datetime.now().isoformat(),
            "model_path": model_path,
            "training_config": {
                "timesteps": training_config.get("timesteps", 100000),
                "batch_size": training_config.get("batch_size", 64),
                "learning_rate": training_config.get("learning_rate", 3e-4),
                "net_arch": training_config.get("net_arch", [64, 64])
            }
        }

        metadata_path = os.path.join(self.models_dir, f"ppo_{symbol}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Training metadata saved for {symbol}")

    def train_agents(self) -> dict:
        """Train RL agents for selected assets using RTX 3090 optimized training"""
        logger.info("Starting RTX 3090 optimized agent training")

        if not self.asset_data:
            logger.error("No asset data loaded, cannot train agents")
            return {}

        # Get configuration
        training_config = self.config.get("training", {})
        timesteps = training_config.get("timesteps", 100000)

        # Check if we should use native PyTorch implementation
        use_native = training_config.get("use_native_pytorch", True)

        # Get available GPUs
        gpu_ids = get_available_gpus()
        if gpu_ids:
            logger.info(f"Found {len(gpu_ids)} GPUs: {gpu_ids}")
            logger.info("Using RTX 3090 optimized training")
        else:
            logger.warning("No GPUs detected, training on CPU")
            use_native = False  # Fall back to stable-baselines3 on CPU

        # Create optimized config file
        import json
        import tempfile
        optimized_config = self.config.copy()
        optimized_config["training"].update({
            "batch_size": 2048,
            "n_steps": 16384,
            "n_epochs": 10,
            "net_arch": [256, 256, 128],
            "use_mixed_precision": True,
            "gradient_accumulation_steps": 4
        })

        # Save temporary config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(optimized_config, f, indent=2)
            temp_config_path = f.name

        # Train agents with optimized implementation
        results = {}
        trained_agents = {}

        try:
            for symbol in self.asset_data.keys():
                # Check if retraining is necessary
                if not self._should_retrain_agent(symbol, training_config):
                    # Use existing model
                    if use_native:
                        model_path = f"{self.models_dir}/native_ppo_{symbol}.pt"
                    else:
                        model_path = f"{self.models_dir}/ppo_{symbol}.zip"

                    if os.path.exists(model_path):
                        trained_agents[symbol] = model_path
                        results[symbol] = {
                            "success": True,
                            "duration": 0.0,
                            "skipped": True,
                            "reason": "Model up-to-date"
                        }
                        logger.info(f"⏭️ {symbol} training skipped - using existing model")
                        continue

                logger.info(f"🚀 Training optimized agent for {symbol}")

                if use_native:
                    # Use native PyTorch for maximum GPU utilization
                    result = train_single_asset_native(
                        symbol=symbol,
                        data_path=self.data_dir,
                        config_path=temp_config_path,
                        output_path=self.models_dir,
                        timesteps=timesteps
                    )
                    model_path = f"{self.models_dir}/native_ppo_{symbol}.pt"
                else:
                    # Use optimized stable-baselines3
                    result = train_single_asset_optimized(
                        symbol=symbol,
                        data_path=self.data_dir,
                        config_path=temp_config_path,
                        output_path=self.models_dir,
                        timesteps=timesteps
                    )
                    model_path = f"{self.models_dir}/ppo_{symbol}.zip"

                results[symbol] = result

                if result.get("success", False):
                    trained_agents[symbol] = model_path
                    # Save metadata for future reference
                    self._save_training_metadata(symbol, training_config, model_path)
                    logger.info(f"✅ {symbol} training completed in {result.get('duration', 0):.2f}s")
                else:
                    logger.error(f"❌ {symbol} training failed: {result.get('error', 'Unknown error')}")

        finally:
            # Clean up temporary config file
            import os
            try:
                os.unlink(temp_config_path)
            except:
                pass

        # Record which agents were trained successfully
        self.trained_agents = trained_agents

        # Log summary
        successful_count = len(trained_agents)
        total_time = sum(r.get('duration', 0) for r in results.values())

        logger.info(f"RTX 3090 optimized training completed:")
        logger.info(f"✅ Successfully trained: {successful_count}/{len(self.asset_data)} agents")
        logger.info(f"⏱️  Total training time: {total_time:.2f} seconds")
        logger.info(f"🚀 Using {'Native PyTorch' if use_native else 'Optimized Stable-Baselines3'}")

        return self.trained_agents

    def generate_trading_signals(self) -> Dict:
        """Generate real-time trading signals from trained models or simple analysis"""
        logger.info("Generating AI trading signals")

        # Get configuration
        training_config = self.config.get("training", {})
        portfolio_balance = training_config.get("initial_balance", 10000.0)

        # Try to use trained models first, fallback to simple signals
        if self.trained_agents:
            logger.info("Using trained AI models for signal generation")
            try:
                # Create signal generator
                signal_generator = TradingSignalGenerator(
                    models_dir=self.models_dir,
                    data_dir=self.data_dir,
                    portfolio_balance=portfolio_balance
                )

                # Generate signals for all trained assets
                asset_list = list(self.trained_agents.keys())
                signals = signal_generator.generate_trading_signals(asset_list)

            except Exception as e:
                logger.warning(f"Error using trained models: {str(e)}")
                logger.info("Falling back to simple momentum-based signals")
                signals = self._generate_simple_signals(portfolio_balance)
        else:
            logger.info("No trained agents available, using simple momentum-based signals")
            signals = self._generate_simple_signals(portfolio_balance)

        # Save signals
        output_path = os.path.join(self.output_dir, "trading_signals.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(signals, f, indent=2)

        # Print summary
        self._print_signals_summary(signals)

        logger.info(f"Generated trading signals for {len(signals.get('signals', {}))} assets")
        logger.info(f"Signals saved to {output_path}")

        return signals

    def _generate_simple_signals(self, portfolio_balance: float) -> Dict:
        """Generate simple momentum-based trading signals"""
        from datetime import datetime

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

        # Use asset data if available, otherwise load top assets
        if self.asset_data:
            assets_to_analyze = list(self.asset_data.keys())
        elif self.selected_assets:
            assets_to_analyze = self.selected_assets
        else:
            # Load top assets from file if available
            try:
                with open(os.path.join(self.output_dir, "top_assets.json"), 'r') as f:
                    top_assets = json.load(f)
                    assets_to_analyze = top_assets["selected_assets"]
            except:
                # Default assets if nothing else is available
                assets_to_analyze = ["BTC_USD", "ETH_USD", "SOL_USD", "LINK_USD", "XRP_USD"]

        logger.info(f"Analyzing {len(assets_to_analyze)} assets: {assets_to_analyze}")

        for symbol in assets_to_analyze:
            try:
                # Get asset data
                if symbol in self.asset_data:
                    df = self.asset_data[symbol]
                else:
                    asset_data = load_and_filter_data(self.data_dir, symbols=[symbol])
                    if symbol not in asset_data:
                        continue
                    df = asset_data[symbol]

                # Calculate simple momentum indicators
                current_price = float(df['close'].iloc[-1])
                prev_price = float(df['close'].iloc[-2])
                change_24h = float((df['close'].iloc[-1] - df['close'].iloc[-25]) / df['close'].iloc[-25] * 100)

                # Simple trading logic
                if change_24h > 3.0:  # Strong upward momentum
                    action = "BUY"
                    confidence = min(0.85, 0.6 + abs(change_24h) / 20.0)
                    position_size = (portfolio_balance * 0.15) / current_price  # 15% of portfolio
                elif change_24h < -3.0:  # Strong downward momentum
                    action = "SELL"
                    confidence = min(0.85, 0.6 + abs(change_24h) / 20.0)
                    position_size = (portfolio_balance * 0.10) / current_price  # 10% of portfolio
                else:  # Sideways movement
                    action = "HOLD"
                    confidence = 0.65
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
                    "reasoning": self._get_simple_reasoning(action, change_24h, current_price, confidence)
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

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
                continue

        return signals

    def _get_simple_reasoning(self, action: str, change_24h: float, price: float, confidence: float) -> str:
        """Generate reasoning for simple signals"""
        if action == "BUY":
            return f"Strong upward momentum (+{change_24h:.1f}% in 24h) with {confidence:.1%} confidence at ${price:.4f}"
        elif action == "SELL":
            return f"Strong downward momentum ({change_24h:.1f}% in 24h) with {confidence:.1%} confidence at ${price:.4f}"
        else:
            return f"Sideways movement ({change_24h:.1f}% in 24h) suggests holding with {confidence:.1%} confidence at ${price:.4f}"

    def _print_signals_summary(self, signals: Dict):
        """Print formatted trading signals summary"""
        print("\n" + "="*80)
        print("🚀 AI TRADING SIGNALS")
        print("="*80)
        print(f"📅 Generated: {signals['timestamp']}")
        print(f"💰 Portfolio Balance: ${signals['portfolio_balance']:,.2f}")
        print(f"📊 Total Signals: {len(signals['signals'])}")
        print()

        for symbol, signal in signals["signals"].items():
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[signal["action"]]

            print(f"{action_emoji} {signal['symbol']}: {signal['action']}")
            print(f"   💵 Price: ${signal['current_price']:,.4f}")
            print(f"   📏 Size: {signal['position_size']:.6f} units")
            print(f"   💲 Value: ${signal['trade_value_usd']:,.2f}")
            print(f"   🎯 Confidence: {signal['confidence']:.1%}")
            print(f"   💭 Reasoning: {signal['reasoning']}")
            print()

        summary = signals["summary"]
        print("📈 SUMMARY:")
        print(f"   🟢 Buy Signals: {summary['total_buy_signals']}")
        print(f"   🔴 Sell Signals: {summary['total_sell_signals']}")
        print(f"   🟡 Hold Signals: {summary['total_hold_signals']}")
        print(f"   💰 Total Trade Value: ${summary['total_trade_value']:,.2f}")
        print("="*80)

    def create_portfolio_allocator(self) -> PortfolioAllocator:
        """Create portfolio allocator"""
        logger.info("Creating portfolio allocator")
        
        # Get configuration
        portfolio_config = self.config.get("portfolio", {})
        
        # Create allocator
        self.portfolio_allocator = PortfolioAllocator(
            mode="sharpe",
            risk_aversion=portfolio_config.get("risk_aversion", 1.2),
            max_allocation=portfolio_config.get("max_allocation", 0.4),
            min_allocation=portfolio_config.get("min_allocation", 0.05),
            method=portfolio_config.get("allocation_method", "risk_parity")
        )
        
        logger.info(f"Portfolio allocator created with method: {portfolio_config.get('allocation_method', 'risk_parity')}")
        return self.portfolio_allocator
    
    def run_backtest(self) -> dict:
        """Run backtest"""
        logger.info("Starting backtest")
        
        if not self.asset_data:
            logger.error("No asset data loaded, cannot run backtest")
            return {}
        
        if not self.trained_agents and not self.config.get("backtest", {}).get("use_traditional", False):
            logger.warning("No trained agents available, falling back to traditional strategies")
            use_traditional = True
        else:
            use_traditional = self.config.get("backtest", {}).get("use_traditional", False)
        
        # Create backtest engine
        system_config = self.config.get("system", {})
        training_config = self.config.get("training", {})
        
        engine = BacktestEngine(
            data=self.asset_data,
            start_date=system_config.get("start_date", None),
            end_date=system_config.get("end_date", None),
            initial_capital=training_config.get("initial_balance", 10000.0),
            transaction_fee=training_config.get("transaction_fee", 0.001)
        )
        
        # Run backtest
        if use_traditional:
            # Use traditional strategy
            backtest_config = self.config.get("backtest", {}).get("traditional", {})
            strategy_name = backtest_config.get("strategy", "sma_crossover")
            
            if strategy_name == "sma_crossover":
                strategy_params = {
                    "short_window": backtest_config.get("short_window", 10),
                    "long_window": backtest_config.get("long_window", 50)
                }
                results = engine.backtest_strategy(sma_crossover_strategy, strategy_params)
            else:
                logger.error(f"Unknown strategy: {strategy_name}")
                return {}
        else:
            # Use RL agents
            results = engine.backtest_rl_agent(self.trained_agents)
        
        # Log results
        if "portfolio" in results:
            portfolio = results["portfolio"]
            logger.info(f"Backtest Results:")
            logger.info(f"  Total Return: {portfolio.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max Drawdown: {portfolio.get('max_drawdown', 0):.2%}")
            logger.info(f"  Total Trades: {portfolio.get('total_trades', 0)}")
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "backtest_results.png")
        engine.plot_results(results, save_path=plot_path)
        
        # Save results
        results_path = os.path.join(self.output_dir, "backtest_results.json")
        try:
            # Convert non-serializable objects
            clean_results = {}
            for k, v in results.items():
                if k == "portfolio":
                    clean_results[k] = {
                        kk: vv for kk, vv in v.items() 
                        if not isinstance(vv, (pd.Series, pd.DataFrame))
                    }
                else:
                    clean_results[k] = {
                        kk: vv for kk, vv in v.items() 
                        if not isinstance(vv, (pd.Series, pd.DataFrame))
                    }
                    
            with open(results_path, "w") as f:
                json.dump(clean_results, f, indent=4)
            
            logger.info(f"Backtest results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
        
        return results
    
    def allocate_portfolio(self) -> dict:
        """Allocate portfolio based on agent predictions"""
        logger.info("Running portfolio allocation")
        
        if not self.asset_data or not self.trained_agents:
            logger.error("Missing required data or trained agents")
            return {}
        
        # Get agent predictions
        agent_predictions = {}
        
        for symbol, model_path in self.trained_agents.items():
            if symbol not in self.asset_data:
                continue
                
            try:
                # Create environment
                env = TradingEnv(df=self.asset_data[symbol])
                
                # Load agent
                agent = PPOTradingAgent(symbol=symbol, env=env)
                agent.load(model_path)
                
                # Get prediction (just to get a signal for allocation)
                obs = env.reset()
                action, _states = agent.predict(obs)
                
                # Convert action to a score
                # 0=hold, 1=buy, 2=sell
                if action == 1:  # Buy
                    score = 0.7  # Bullish
                elif action == 2:  # Sell
                    score = 0.3  # Bearish
                else:  # Hold
                    score = 0.5  # Neutral
                
                agent_predictions[symbol] = score
                
            except Exception as e:
                logger.error(f"Error getting prediction for {symbol}: {str(e)}")
        
        # Initialize portfolio allocator if needed
        if self.portfolio_allocator is None:
            self.create_portfolio_allocator()
        
        # Calculate historical returns
        returns_history = {}
        for symbol, df in self.asset_data.items():
            returns_history[symbol] = df['close'].pct_change().fillna(0)
        
        # Allocate portfolio
        allocation = self.portfolio_allocator.allocate_portfolio(
            agent_predictions=agent_predictions,
            returns_history=returns_history,
            allocation_date=datetime.now().isoformat()
        )
        
        # Log allocation
        logger.info("Portfolio Allocation:")
        for symbol, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {symbol}: {weight:.2%}")
        
        # Save allocation
        allocation_path = os.path.join(self.output_dir, "portfolio_allocation.json")
        with open(allocation_path, "w") as f:
            json.dump({
                "date": datetime.now().isoformat(),
                "allocation": {k: float(v) for k, v in allocation.items()}
            }, f, indent=4)
        
        logger.info(f"Portfolio allocation saved to {allocation_path}")
        return allocation
    
    def generate_report(self, results: dict) -> str:
        """Generate report using LLM reasoning engine"""
        logger.info("Generating trading report")
        
        try:
            # Initialize reasoning engine
            reasoning_engine = ReasoningEngine()
            
            # Generate report
            report = reasoning_engine.generate_trading_report(
                asset_data=self.asset_data,
                agent_predictions={s: "bullish" if self.trained_agents.get(s) else "unknown" for s in self.selected_assets},
                portfolio_allocation=self.portfolio_allocator.historical_allocations[-1] if self.portfolio_allocator else {},
                backtest_results=results
            )
            
            # Save report
            report_path = os.path.join(self.output_dir, "trading_report.md")
            with open(report_path, "w") as f:
                f.write(report)
                
            logger.info(f"Trading report saved to {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return "Error generating report"
    
    def run_full_pipeline(self):
        """Run the full trading system pipeline"""
        logger.info("Starting full trading system pipeline")

        # Select assets
        self.select_assets()

        # Load data
        self.load_data()

        # Train agents
        self.train_agents()

        # Generate trading signals
        self.generate_trading_signals()

        # Run backtest
        backtest_results = self.run_backtest()

        # Allocate portfolio
        self.allocate_portfolio()

        # Generate report
        self.generate_report(backtest_results)

        logger.info("Trading system pipeline completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Trading System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for market data")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for output files")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for model files")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--pipeline", action="store_true", help="Run the full pipeline")
    parser.add_argument("--select-assets", action="store_true", help="Run asset selection only")
    parser.add_argument("--train", action="store_true", help="Run agent training only")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if models exist")
    parser.add_argument("--signals", action="store_true", help="Generate trading signals only")
    parser.add_argument("--backtest", action="store_true", help="Run backtest only")
    parser.add_argument("--allocate", action="store_true", help="Run portfolio allocation only")
    
    args = parser.parse_args()
    
    # Initialize trading system
    system = TradingSystem(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir
    )

    # Set force retrain flag if specified
    if hasattr(args, 'force_retrain') and args.force_retrain:
        system.force_retrain = True
        logger.info("Force retraining enabled - will retrain all models")
    
    # Execute selected actions
    if args.pipeline:
        system.run_full_pipeline()
    else:
        # Run individual components if specified
        if args.select_assets:
            system.select_assets()
            
        if args.select_assets or args.train or args.backtest or args.allocate:
            system.load_data()
            
        if args.train:
            system.train_agents()

        if args.signals:
            system.generate_trading_signals()

        if args.backtest:
            system.run_backtest()

        if args.allocate:
            system.allocate_portfolio()

if __name__ == "__main__":
    main() 