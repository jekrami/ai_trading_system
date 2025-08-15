import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import torch

# Import our custom environment
from envs.trading_env import TradingEnv
#from ai_trading_system.envs.trading_env import TradingEnv 

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.portfolio_values = []
        self.returns = []
        self.drawdowns = []
        self.epsilon = 1e-10  # Small epsilon to prevent division by zero

    def _on_step(self) -> bool:
        # Extract info from the environment
        env = self.model.get_env().envs[0].unwrapped
        portfolio_value = max(env._calculate_portfolio_value(), self.epsilon)  # Ensure positive value
        
        # Store values for tracking
        self.portfolio_values.append(portfolio_value)
        
        # Calculate returns if we have enough data
        if len(self.portfolio_values) > 1:
            prev_value = max(self.portfolio_values[-2], self.epsilon)  # Ensure positive value
            daily_return = (portfolio_value / prev_value) - 1
            self.returns.append(daily_return)
            
            # Calculate drawdown
            peak = max(self.portfolio_values)
            drawdown = (portfolio_value - peak) / (peak + self.epsilon)
            self.drawdowns.append(drawdown)
            
            # Log metrics
            self.logger.record('trading/portfolio_value', portfolio_value)
            self.logger.record('trading/daily_return', daily_return)
            self.logger.record('trading/drawdown', drawdown)
            
            # Calculate Sharpe ratio with at least 10 data points
            if len(self.returns) >= 10:
                std_returns = np.std(self.returns) + self.epsilon  # Add epsilon to prevent division by zero
                sharpe = np.mean(self.returns) / std_returns * np.sqrt(252)  # Annualized
                self.logger.record('trading/sharpe_ratio', sharpe)
                
            # Log maximum drawdown
            max_dd = min(self.drawdowns)
            self.logger.record('trading/max_drawdown', max_dd)
        
        return True

class PPOTradingAgent:
    """
    PPO agent for cryptocurrency trading using stable-baselines3.
    """
    def __init__(
        self,
        symbol: str,
        env: TradingEnv,
        model_path: str = "models",
        tensorboard_log: str = "logs",
        device: Union[str, torch.device] = "auto",
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        policy_kwargs: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize a PPO trading agent.
        
        Args:
            symbol: Asset symbol this agent will trade
            env: Trading environment
            model_path: Directory to save models
            tensorboard_log: Directory for tensorboard logs
            device: Device to run the model on ('auto', 'cuda', 'cuda:0', 'cpu')
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size
            n_epochs: Number of epoch when optimizing the surrogate loss
            gamma: Discount factor
            learning_rate: Learning rate
            policy_kwargs: Arguments to pass to the policy
            seed: Random seed
        """
        self.symbol = symbol
        self.env = env
        self.model_path = os.path.join(model_path, f"ppo_{symbol}")
        self.tensorboard_log = os.path.join(tensorboard_log, symbol)
        
        # Create directories if they don't exist
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(self.tensorboard_log, exist_ok=True)
        
        # Set random seed if provided
        if seed is not None:
            set_random_seed(seed)
        
        # Set up policy_kwargs if not provided
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [64, 64],  # Simpler network architecture
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True,  # Use orthogonal initialization for stability
                "log_std_init": -2.0,  # Smaller initial log std for more conservative exploration
            }
        
        # Build PPO agent
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=self.tensorboard_log,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            device=device,
            ent_coef=0.01,  # Add entropy coefficient for exploration
            clip_range=0.2,  # Standard PPO clipping parameter
            max_grad_norm=0.5,  # Clip gradients for stability
            gae_lambda=0.95,  # GAE parameter
            normalize_advantage=True  # Normalize advantage estimates
        )
    
    def train(self, total_timesteps: int = 100000, callback=None):
        """Train the agent"""
        logger.info(f"Starting training of {self.symbol} agent for {total_timesteps} timesteps")
        
        # Set up callbacks
        callbacks = []
        if callback:
            callbacks.append(callback)
        
        # Add custom tensorboard callback
        tb_callback = TensorboardCallback()
        callbacks.append(tb_callback)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=self.symbol
        )
        
        # Save the model
        self.save()
        
        logger.info(f"Training of {self.symbol} agent completed")
    
    def save(self, path: Optional[str] = None):
        """Save the model"""
        save_path = path or f"{self.model_path}.zip"
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Save a metadata file with the model
        metadata = {
            "symbol": self.symbol,
            "train_date": pd.Timestamp.now().isoformat(),
            "model_type": "PPO",
            "observation_space": str(self.env.observation_space),
            "action_space": str(self.env.action_space)
        }
        
        with open(f"{self.model_path}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
    
    def load(self, path: Optional[str] = None):
        """Load the model"""
        load_path = path or f"{self.model_path}.zip"
        if os.path.exists(load_path):
            self.model = PPO.load(load_path, env=self.env)
            logger.info(f"Model loaded from {load_path}")
        else:
            logger.error(f"Model file {load_path} not found")
    
    def predict(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        """Make a prediction for a single observation"""
        action, _states = self.model.predict(observation, deterministic=True)
        return action, _states
    
    def backtest(self, test_env: TradingEnv) -> Dict:
        """Run a backtest on the test environment"""
        logger.info(f"Starting backtest for {self.symbol}")
        
        obs = test_env.reset()
        done = False
        info_history = []
        
        while not done:
            action, _states = self.predict(obs)
            obs, reward, done, info = test_env.step(action)
            info_history.append(info)
        
        # Get trade history
        trade_history = test_env.get_trade_history()
        
        # Calculate metrics
        portfolio_values = [info['portfolio_value'] for info in info_history]
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value / initial_value) - 1
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            returns.append(daily_return)
        
        # Calculate metrics
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)  # Annualized
        
        # Calculate drawdowns
        drawdowns = []
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak
            drawdowns.append(drawdown)
        
        max_drawdown = min(drawdowns)
        
        # Prepare results
        results = {
            "symbol": self.symbol,
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_trades": test_env.total_trades,
            "total_fee_paid": test_env.total_fee_paid
        }
        
        logger.info(f"Backtest results for {self.symbol}: {results}")
        
        return {"results": results, "trade_history": trade_history}

def make_env(env_class, **kwargs):
    """Helper function to create a vectorized environment"""
    def _init():
        env = env_class(**kwargs)
        env = Monitor(env)
        return env
    return _init

def create_vectorized_env(env_class, num_envs=4, **kwargs):
    """Create a vectorized environment for parallel training"""
    env_fns = [make_env(env_class, **kwargs) for _ in range(num_envs)]
    return SubprocVecEnv(env_fns)

def setup_multi_gpu_training(symbols: List[str], 
                            dataframes: Dict[str, pd.DataFrame], 
                            gpu_ids: List[int],
                            **env_params):
    """Setup multi-GPU training for multiple assets"""
    agents = []
    
    for i, symbol in enumerate(symbols):
        # Determine which GPU to use (round-robin)
        gpu_id = gpu_ids[i % len(gpu_ids)]
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        # Create environment for this asset
        env = TradingEnv(df=dataframes[symbol], **env_params)
        
        # Create agent
        agent = PPOTradingAgent(
            symbol=symbol,
            env=env,
            device=device
        )
        
        agents.append((agent, device))
    
    return agents 