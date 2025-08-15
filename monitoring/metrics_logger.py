import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import psutil
import socket
import platform

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Logs and tracks system and trading metrics for monitoring.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        metrics_file: str = "metrics.json",
        log_interval: int = 3600,  # 1 hour default
        enable_prometheus: bool = False,
        prometheus_port: int = 8000,
        system_metrics: bool = True
    ):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory for log files
            metrics_file: File name for metrics JSON
            log_interval: Interval between logs in seconds
            enable_prometheus: Whether to enable Prometheus endpoint
            prometheus_port: Port for Prometheus server
            system_metrics: Whether to log system metrics
        """
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, metrics_file)
        self.log_interval = log_interval
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.system_metrics = system_metrics
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = {
            "system": [],
            "portfolio": [],
            "assets": {},
            "agents": {}
        }
        
        # Load existing metrics if available
        self._load_metrics()
        
        # Initialize Prometheus server if enabled
        if enable_prometheus:
            try:
                self._initialize_prometheus_server()
            except ImportError:
                logger.error("Could not initialize Prometheus server. Make sure prometheus_client is installed.")
                self.enable_prometheus = False
    
    def _load_metrics(self):
        """Load metrics from file if it exists"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
                logger.info(f"Loaded metrics from {self.metrics_file}")
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.debug(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def _initialize_prometheus_server(self):
        """Initialize Prometheus server"""
        from prometheus_client import start_http_server, Gauge, Counter
        
        # Start Prometheus server
        start_http_server(self.prometheus_port)
        logger.info(f"Started Prometheus server on port {self.prometheus_port}")
        
        # Define Prometheus metrics
        self.prom_portfolio_value = Gauge('portfolio_value', 'Current portfolio value')
        self.prom_portfolio_return = Gauge('portfolio_return', 'Portfolio return')
        self.prom_trade_count = Counter('trade_count', 'Number of trades')
        self.prom_asset_prices = {}  # Will be populated as assets are added
    
    def log_system_metrics(self):
        """Log system metrics (CPU, memory, disk, etc.)"""
        if not self.system_metrics:
            return
            
        try:
            # Get current time
            now = datetime.now().isoformat()
            
            # Collect metrics
            metrics = {
                "timestamp": now,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "hostname": socket.gethostname(),
                "platform": platform.platform()
            }
            
            # Add GPU metrics if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                gpu_metrics = []
                
                for i, gpu in enumerate(gpus):
                    gpu_metrics.append({
                        "id": i,
                        "name": gpu.name,
                        "load": gpu.load,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    })
                
                metrics["gpus"] = gpu_metrics
            except ImportError:
                logger.debug("GPUtil not installed, skipping GPU metrics")
            
            # Log metrics
            self.metrics_history["system"].append(metrics)
            logger.debug(f"Logged system metrics: {metrics}")
            
            # Save metrics periodically
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error logging system metrics: {str(e)}")
    
    def log_portfolio_metrics(
        self,
        portfolio_value: float,
        allocation: Dict[str, float],
        returns: Dict[str, float],
        realized_pnl: float,
        unrealized_pnl: float
    ):
        """
        Log portfolio metrics.
        
        Args:
            portfolio_value: Current portfolio value
            allocation: Current asset allocation
            returns: Dictionary mapping assets to returns
            realized_pnl: Realized profit and loss
            unrealized_pnl: Unrealized profit and loss
        """
        try:
            # Get current time
            now = datetime.now().isoformat()
            
            # Collect metrics
            metrics = {
                "timestamp": now,
                "portfolio_value": portfolio_value,
                "allocation": allocation,
                "returns": returns,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl
            }
            
            # Log metrics
            self.metrics_history["portfolio"].append(metrics)
            logger.debug(f"Logged portfolio metrics: {metrics}")
            
            # Update Prometheus metrics if enabled
            if self.enable_prometheus:
                self.prom_portfolio_value.set(portfolio_value)
                # Calculate daily return
                if len(self.metrics_history["portfolio"]) > 1:
                    prev_value = self.metrics_history["portfolio"][-2]["portfolio_value"]
                    daily_return = (portfolio_value / prev_value) - 1
                    self.prom_portfolio_return.set(daily_return)
            
            # Save metrics periodically
            if len(self.metrics_history["portfolio"]) % 5 == 0:
                self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error logging portfolio metrics: {str(e)}")
    
    def log_asset_metrics(
        self,
        symbol: str,
        price: float,
        volume: float,
        position: float,
        cost_basis: float,
        unrealized_pnl: float,
        realized_pnl: float,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log asset-specific metrics.
        
        Args:
            symbol: Asset symbol
            price: Current price
            volume: Trading volume
            position: Current position size
            cost_basis: Cost basis for position
            unrealized_pnl: Unrealized profit and loss
            realized_pnl: Realized profit and loss
            additional_metrics: Additional metrics to log
        """
        try:
            # Get current time
            now = datetime.now().isoformat()
            
            # Initialize asset in history if needed
            if symbol not in self.metrics_history["assets"]:
                self.metrics_history["assets"][symbol] = []
                
                # Create Prometheus gauge if enabled
                if self.enable_prometheus:
                    self.prom_asset_prices[symbol] = Gauge(
                        f'asset_price_{symbol.lower()}', 
                        f'Price of {symbol}'
                    )
            
            # Collect metrics
            metrics = {
                "timestamp": now,
                "price": price,
                "volume": volume,
                "position": position,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl
            }
            
            # Add additional metrics if provided
            if additional_metrics:
                metrics.update(additional_metrics)
            
            # Log metrics
            self.metrics_history["assets"][symbol].append(metrics)
            logger.debug(f"Logged metrics for {symbol}: {metrics}")
            
            # Update Prometheus metrics if enabled
            if self.enable_prometheus and symbol in self.prom_asset_prices:
                self.prom_asset_prices[symbol].set(price)
            
            # Save metrics periodically (every 10 updates per asset)
            asset_count = len(self.metrics_history["assets"][symbol])
            if asset_count % 10 == 0:
                self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error logging asset metrics for {symbol}: {str(e)}")
    
    def log_agent_metrics(
        self,
        agent_id: str,
        action: str,
        confidence: float,
        reward: float,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log agent-specific metrics.
        
        Args:
            agent_id: Agent identifier
            action: Action taken
            confidence: Confidence in action
            reward: Reward received
            metrics: Additional metrics
        """
        try:
            # Get current time
            now = datetime.now().isoformat()
            
            # Initialize agent in history if needed
            if agent_id not in self.metrics_history["agents"]:
                self.metrics_history["agents"][agent_id] = []
            
            # Collect metrics
            agent_metrics = {
                "timestamp": now,
                "action": action,
                "confidence": confidence,
                "reward": reward
            }
            
            # Add additional metrics if provided
            if metrics:
                agent_metrics.update(metrics)
            
            # Log metrics
            self.metrics_history["agents"][agent_id].append(agent_metrics)
            logger.debug(f"Logged metrics for agent {agent_id}: {agent_metrics}")
            
            # Save metrics periodically
            agent_count = len(self.metrics_history["agents"][agent_id])
            if agent_count % 10 == 0:
                self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error logging agent metrics for {agent_id}: {str(e)}")
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        timestamp: Optional[str] = None,
        trade_id: Optional[str] = None,
        fees: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a trade.
        
        Args:
            symbol: Asset symbol
            action: Trade action (buy, sell)
            price: Trade price
            quantity: Trade quantity
            timestamp: Trade timestamp
            trade_id: Trade ID
            fees: Trading fees
            metadata: Additional metadata
        """
        try:
            # Get current time if not provided
            if not timestamp:
                timestamp = datetime.now().isoformat()
                
            # Generate trade ID if not provided
            if not trade_id:
                trade_id = f"{symbol}_{action}_{int(time.time())}"
            
            # Collect trade info
            trade = {
                "trade_id": trade_id,
                "timestamp": timestamp,
                "symbol": symbol,
                "action": action,
                "price": price,
                "quantity": quantity,
                "fees": fees,
                "value": price * quantity
            }
            
            # Add metadata if provided
            if metadata:
                trade["metadata"] = metadata
                
            # Initialize asset if needed
            if symbol not in self.metrics_history["assets"]:
                self.metrics_history["assets"][symbol] = []
            
            # Find latest asset metrics
            latest_metrics = {}
            if self.metrics_history["assets"][symbol]:
                latest_metrics = self.metrics_history["assets"][symbol][-1].copy()
                latest_metrics["timestamp"] = timestamp
                
                # Update trade count in latest metrics
                if "trade_count" in latest_metrics:
                    latest_metrics["trade_count"] += 1
                else:
                    latest_metrics["trade_count"] = 1
                    
                # Add trade to metrics
                latest_metrics["last_trade"] = {
                    "action": action,
                    "price": price,
                    "quantity": quantity,
                    "timestamp": timestamp
                }
                
                # Append updated metrics
                self.metrics_history["assets"][symbol].append(latest_metrics)
                
            # Update Prometheus metrics if enabled
            if self.enable_prometheus:
                self.prom_trade_count.inc()
            
            # Log trade
            logger.info(f"Logged trade: {trade}")
            
            # Save metrics
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error logging trade for {symbol}: {str(e)}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of portfolio performance"""
        if not self.metrics_history["portfolio"]:
            return {}
            
        # Get portfolio history
        history = self.metrics_history["portfolio"]
        
        # Calculate metrics
        portfolio_values = [entry["portfolio_value"] for entry in history]
        latest_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        
        # Calculate returns
        total_return = (latest_value / initial_value) - 1
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            returns.append(daily_return)
            
        # Calculate metrics
        sharpe = 0
        volatility = 0
        max_drawdown = 0
        
        if returns:
            returns_array = np.array(returns)
            sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-10) * np.sqrt(252)
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # Calculate drawdown
            cumulative = np.cumprod(1 + returns_array)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_drawdown = np.min(drawdown)
        
        # Get latest allocation
        allocation = history[-1].get("allocation", {})
        
        # Prepare summary
        summary = {
            "current_value": latest_value,
            "initial_value": initial_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "latest_allocation": allocation
        }
        
        return summary
    
    def get_asset_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary for a specific asset"""
        if symbol not in self.metrics_history["assets"] or not self.metrics_history["assets"][symbol]:
            return {}
            
        # Get asset history
        history = self.metrics_history["assets"][symbol]
        
        # Get latest entry
        latest = history[-1]
        
        # Calculate metrics if we have enough data
        price_history = [entry["price"] for entry in history]
        
        metrics = {
            "current_price": latest["price"],
            "position": latest.get("position", 0),
            "unrealized_pnl": latest.get("unrealized_pnl", 0),
            "realized_pnl": latest.get("realized_pnl", 0)
        }
        
        # Calculate additional metrics if we have enough data
        if len(price_history) > 1:
            # Price changes
            daily_change = (price_history[-1] / price_history[-2]) - 1
            
            # Longer-term changes if available
            if len(price_history) >= 7:
                weekly_change = (price_history[-1] / price_history[-7]) - 1
                metrics["weekly_change"] = weekly_change
                
            if len(price_history) >= 30:
                monthly_change = (price_history[-1] / price_history[-30]) - 1
                metrics["monthly_change"] = monthly_change
            
            metrics["daily_change"] = daily_change
            
        return metrics
    
    def export_metrics_csv(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export metrics to CSV files.
        
        Args:
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping metric types to file paths
        """
        if output_dir is None:
            output_dir = self.log_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        export_files = {}
        
        try:
            # Export system metrics
            if self.metrics_history["system"]:
                system_df = pd.DataFrame(self.metrics_history["system"])
                system_file = os.path.join(output_dir, "system_metrics.csv")
                system_df.to_csv(system_file, index=False)
                export_files["system"] = system_file
                
            # Export portfolio metrics
            if self.metrics_history["portfolio"]:
                # Need to handle nested dictionaries
                portfolio_data = []
                for entry in self.metrics_history["portfolio"]:
                    data = {
                        "timestamp": entry["timestamp"],
                        "portfolio_value": entry["portfolio_value"],
                        "realized_pnl": entry.get("realized_pnl", 0),
                        "unrealized_pnl": entry.get("unrealized_pnl", 0)
                    }
                    portfolio_data.append(data)
                
                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_file = os.path.join(output_dir, "portfolio_metrics.csv")
                portfolio_df.to_csv(portfolio_file, index=False)
                export_files["portfolio"] = portfolio_file
                
            # Export asset metrics
            for symbol, history in self.metrics_history["assets"].items():
                if history:
                    asset_df = pd.DataFrame(history)
                    asset_file = os.path.join(output_dir, f"asset_{symbol}_metrics.csv")
                    asset_df.to_csv(asset_file, index=False)
                    export_files[f"asset_{symbol}"] = asset_file
                    
            logger.info(f"Exported metrics to {output_dir}")
            return export_files
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return export_files

if __name__ == "__main__":
    # Example usage
    metrics_logger = MetricsLogger()
    
    # Log system metrics
    metrics_logger.log_system_metrics()
    
    # Log portfolio metrics
    metrics_logger.log_portfolio_metrics(
        portfolio_value=10500.0,
        allocation={"BTC": 0.6, "ETH": 0.3, "SOL": 0.1},
        returns={"BTC": 0.05, "ETH": -0.02, "SOL": 0.08},
        realized_pnl=120.0,
        unrealized_pnl=380.0
    )
    
    # Log asset metrics
    metrics_logger.log_asset_metrics(
        symbol="BTC",
        price=50000.0,
        volume=2.5,
        position=0.12,
        cost_basis=48000.0,
        unrealized_pnl=240.0,
        realized_pnl=0.0
    )
    
    print("Metrics logged successfully") 