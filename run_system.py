#!/usr/bin/env python
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import system components
from agents.asset_universe import AssetUniverseSelector, load_and_filter_data
from agents.ppo_agent import PPOTradingAgent
from agents.multi_gpu_trainer import train_multiple_agents, get_available_gpus
from envs.trading_env import TradingEnv
from meta_agent.portfolio_allocator import PortfolioAllocator
from backtesting.backtest_engine import BacktestEngine, sma_crossover_strategy
from monitoring.metrics_logger import MetricsLogger
from llm.reasoning_engine import ReasoningEngine

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
    
    def train_agents(self) -> dict:
        """Train RL agents for selected assets"""
        logger.info("Starting agent training")
        
        if not self.asset_data:
            logger.error("No asset data loaded, cannot train agents")
            return {}
        
        # Get configuration
        training_config = self.config.get("training", {})
        
        # Get available GPUs
        gpu_ids = get_available_gpus()
        if gpu_ids:
            logger.info(f"Found {len(gpu_ids)} GPUs: {gpu_ids}")
        else:
            logger.warning("No GPUs detected, training on CPU")
        
        # Train agents
        results = train_multiple_agents(
            asset_list=list(self.asset_data.keys()),
            data_path=self.data_dir,
            output_path=self.models_dir,
            gpu_ids=gpu_ids,
            timesteps=training_config.get("timesteps", 100000)
        )
        
        # Record which agents were trained successfully
        self.trained_agents = {
            symbol: f"{self.models_dir}/ppo_{symbol}.zip" 
            for symbol, result in results.items() 
            if result.get("success", False)
        }
        
        logger.info(f"Successfully trained {len(self.trained_agents)} agents")
        return self.trained_agents
    
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
            
        if args.backtest:
            system.run_backtest()
            
        if args.allocate:
            system.allocate_portfolio()

if __name__ == "__main__":
    main() 