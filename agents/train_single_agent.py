import os
import sys
import argparse
import logging
import json
from datetime import datetime
import pandas as pd

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from agents.ppo_agent import PPOTradingAgent
from agents.ppo_agent import PPOTradingAgent
from agents.asset_universe import load_and_filter_data
from envs.trading_env import TradingEnv

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train a single PPO agent for a given symbol.")
    parser.add_argument('--symbol', type=str, required=True, help='Asset symbol (e.g., BTC_USD)')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the model')
    parser.add_argument('--timesteps', type=int, default=100000, help='Number of training timesteps')
    parser.add_argument('--config-path', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--log-path', type=str, default=None, help='Path to save logs (optional)')
    args = parser.parse_args()

    # Set up file logger if log_path is provided
    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
        log_file = os.path.join(args.log_path, f"{args.symbol}_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")

    # Load config if provided
    config = {}
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {args.config_path}")
    else:
        logger.info("No config file provided or found, using defaults.")

    # Training parameters (use config if available, else defaults)
    training_cfg = config.get('training', {}) if config else {}
    window_size = training_cfg.get('window_size', 24)
    initial_balance = training_cfg.get('initial_balance', 10000.0)
    transaction_fee = training_cfg.get('transaction_fee', 0.001)
    batch_size = training_cfg.get('batch_size', 64)
    learning_rate = training_cfg.get('learning_rate', 3e-4)
    n_steps = training_cfg.get('n_steps', 2048)
    n_epochs = training_cfg.get('n_epochs', 10)
    gamma = training_cfg.get('gamma', 0.99)

    # Load data
    logger.info(f"Loading data for {args.symbol} from {args.data_path}")
    asset_data = load_and_filter_data(
        data_dir=args.data_path,
        symbols=[args.symbol],
        start_date=training_cfg.get('start_date'),
        end_date=training_cfg.get('end_date')
    )
    if args.symbol not in asset_data:
        logger.error(f"No data found for symbol {args.symbol}. Exiting.")
        sys.exit(1)
    df = asset_data[args.symbol]
    
    # Check for NaN values in the data
    if df.isna().any().any():
        logger.warning(f"Found NaN values in the data. Filling with forward fill method.")
        df = df.ffill().bfill()  # Forward fill then backward fill any remaining NaNs
        
    # Log data statistics
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns.tolist()}")
    logger.info(f"Data sample: \n{df.head(3)}")

    # Create environment
    env = TradingEnv(
        df=df,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee
    )

    # Create agent
    agent = PPOTradingAgent(
        symbol=args.symbol,
        env=env,
        model_path=args.output_path,
        tensorboard_log=args.log_path or 'logs',
        device='cuda',
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        learning_rate=learning_rate
    )

    # Train agent
    logger.info(f"Starting training for {args.symbol} for {args.timesteps} timesteps")
    agent.train(total_timesteps=args.timesteps)
    logger.info(f"Training completed for {args.symbol}")

if __name__ == '__main__':
    main() 