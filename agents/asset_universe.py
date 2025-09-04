import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import glob
from datetime import datetime, timedelta

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetUniverseSelector:
    """
    Selects the best assets to trade based on various criteria.
    Uses a combination of market data, technical indicators, and ML.
    """
    def __init__(
        self, 
        data_dir: str,
        lookback_days: int = 30,
        max_assets: int = 10,
        min_volume_usd: float = 1000000,  # Minimum daily USD volume
        min_market_cap_usd: float = 10000000,  # Minimum market cap
        metric_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the asset universe selector.
        
        Args:
            data_dir: Directory containing market data (CSV files)
            lookback_days: Number of days to look back for metrics
            max_assets: Maximum number of assets to select
            min_volume_usd: Minimum average daily volume in USD
            min_market_cap_usd: Minimum market cap in USD
            metric_weights: Dictionary of metric weights for scoring
        """
        self.data_dir = data_dir
        self.lookback_days = lookback_days
        self.max_assets = max_assets
        self.min_volume_usd = min_volume_usd
        self.min_market_cap_usd = min_market_cap_usd
        
        # Default metric weights
        if metric_weights is None:
            self.metric_weights = {
                "volume": 0.3,
                "volatility": 0.2,
                "trend_strength": 0.2,
                "liquidity": 0.15,
                "market_cap": 0.15
            }
        else:
            self.metric_weights = metric_weights
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data from CSV files in the data directory"""
        logger.info(f"Loading market data from {self.data_dir}")

        # Look for CSV files in multiple locations
        csv_files = []
        # First try the main data directory
        csv_files.extend(glob.glob(os.path.join(self.data_dir, '*_1h.csv')))
        # Then try the 1h subdirectory
        csv_files.extend(glob.glob(os.path.join(self.data_dir, '1h', '*_1h.csv')))
        # Also try files without _1h suffix in main directory
        csv_files.extend(glob.glob(os.path.join(self.data_dir, '*.csv')))

        logger.info(f"Found {len(csv_files)} CSV files")
        asset_data = {}
        
        for file_path in csv_files:
            symbol = os.path.splitext(os.path.basename(file_path))[0].upper().replace('_1H', '')
            try:
                df = pd.read_csv(file_path, skiprows=[1,2])
                df.columns = ['datetime', 'close', 'high', 'low', 'open', 'volume']
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
                df = df.sort_index()
                
                # Calculate derived metrics
                # Approximate market cap as (price * volume / 30) * 30 assuming average daily trading is ~3% of market cap
                df['market_cap_proxy'] = df['close'] * df['volume'] 
                
                asset_data[symbol] = df
                logger.info(f"Loaded data for {symbol}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {str(e)}")
                continue
                
        return asset_data
    
    def calculate_asset_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics for a single asset"""
        # Filter to lookback period
        cutoff_date = df.index.max() - timedelta(days=self.lookback_days)
        df_period = df[df.index >= cutoff_date]
        
        if len(df_period) < 5:  # Require at least 5 data points
            return None
        
        metrics = {}
        
        # Trading volume (in USD)
        metrics["volume"] = df_period['volume'].mean() * df_period['close'].mean()
        
        # Volatility (daily)
        hourly_returns = df_period['close'].pct_change().dropna()
        metrics["volatility"] = hourly_returns.std() * np.sqrt(24)  # Scale to daily
        
        # Trend strength - absolute value of the correlation coefficient of price vs. time
        # Higher abs value means stronger trend (up or down)
        time_index = np.arange(len(df_period))
        price_series = df_period['close'].values
        if len(time_index) > 1:  # Need at least 2 points for correlation
            correlation = np.corrcoef(time_index, price_series)[0, 1]
            metrics["trend_strength"] = abs(correlation)
        else:
            metrics["trend_strength"] = 0
            
        # Liquidity - average spread approximation (high-low)/close
        metrics["liquidity"] = 1 - df_period[['high', 'low', 'close']].apply(
            lambda x: (x['high'] - x['low']) / x['close'], axis=1
        ).mean()
        
        # Market capitalization proxy
        metrics["market_cap"] = df_period['market_cap_proxy'].mean()
        
        # Latest price
        metrics["current_price"] = df_period['close'].iloc[-1]
        
        # Latest market cap
        metrics["current_market_cap"] = df_period['market_cap_proxy'].iloc[-1]
        
        return metrics
    
    def score_assets(self, metrics_dict: Dict[str, Dict[str, float]]) -> pd.Series:
        """Score assets based on their metrics"""
        if not metrics_dict:
            return pd.Series()
            
        # Create DataFrame from metrics
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        # Apply minimum volume and market cap filters
        filtered_df = metrics_df[
            (metrics_df['volume'] >= self.min_volume_usd) & 
            (metrics_df['market_cap'] >= self.min_market_cap_usd)
        ]
        
        if len(filtered_df) == 0:
            logger.warning("No assets passed the minimum filters")
            return pd.Series()
        
        # Normalize metrics for scoring
        scaler = StandardScaler()
        score_metrics = ['volume', 'volatility', 'trend_strength', 'liquidity', 'market_cap']
        
        if len(filtered_df) < 2:  # Need at least 2 samples for StandardScaler
            # If only one asset passed filters, return it with score 1.0
            return pd.Series({filtered_df.index[0]: 1.0})
            
        normalized_metrics = pd.DataFrame(
            scaler.fit_transform(filtered_df[score_metrics]),
            index=filtered_df.index,
            columns=score_metrics
        )
        
        # Calculate weighted score
        scores = pd.Series(0.0, index=normalized_metrics.index)
        for metric, weight in self.metric_weights.items():
            if metric in normalized_metrics.columns:
                scores += normalized_metrics[metric] * weight
        
        # Sort by score (descending)
        return scores.sort_values(ascending=False)
    
    def select_assets(self) -> Dict:
        """Select the best assets to trade"""
        # Load market data
        asset_data = self.load_market_data()
        
        if not asset_data:
            logger.error("No asset data available")
            return {"selected_assets": [], "metrics": {}, "scores": {}}
        
        # Calculate metrics for each asset
        asset_metrics = {}
        for symbol, df in asset_data.items():
            metrics = self.calculate_asset_metrics(df)
            if metrics is not None:
                asset_metrics[symbol] = metrics
        
        # Score assets
        scores = self.score_assets(asset_metrics)
        
        # Select top N assets
        selected_assets = list(scores.head(self.max_assets).index)
        
        # Prepare detailed metrics for output
        detailed_metrics = {}
        for symbol in selected_assets:
            if symbol in asset_metrics:
                detailed_metrics[symbol] = asset_metrics[symbol]
        
        # Prepare scores for output
        score_dict = {}
        for symbol in selected_assets:
            if symbol in scores:
                score_dict[symbol] = float(scores[symbol])
        
        result = {
            "selection_date": datetime.now().isoformat(),
            "selected_assets": selected_assets,
            "metrics": detailed_metrics,
            "scores": score_dict
        }
        
        # Save the result to a JSON file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "top_assets.json")
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        
        logger.info(f"Selected {len(selected_assets)} assets: {', '.join(selected_assets)}")
        logger.info(f"Asset selection saved to {output_file}")
        
        return result


def load_and_filter_data(
    data_dir: str,
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load data for a list of symbols, with optional date filtering.
    
    Args:
        data_dir: Directory containing CSV data files
        symbols: List of symbols to load. If None, load all available.
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if symbols is None:
        # Load all available assets from multiple locations
        csv_files = []
        csv_files.extend(glob.glob(os.path.join(data_dir, '*_1h.csv')))
        csv_files.extend(glob.glob(os.path.join(data_dir, '1h', '*_1h.csv')))
        symbols = [os.path.splitext(os.path.basename(file))[0].upper().replace('_1H', '')
                  for file in csv_files]

    asset_data = {}

    for symbol in symbols:
        # Try multiple file path patterns
        file_paths = [
            os.path.join(data_dir, f"{symbol.lower()}_1h.csv"),
            os.path.join(data_dir, '1h', f"{symbol.upper()}_1h.csv"),
            os.path.join(data_dir, f"{symbol.upper()}.csv")
        ]

        file_path = None
        for path in file_paths:
            if os.path.exists(path):
                file_path = path
                break

        if file_path is None:
            logger.warning(f"Data file for {symbol} not found in any of: {file_paths}")
            continue
            
        try:
            df = pd.read_csv(file_path, skiprows=[1,2])
            df.columns = ['datetime', 'close', 'high', 'low', 'open', 'volume']
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df = df.sort_index()
            
            # Apply date filters if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            # Make sure we have enough data
            if len(df) < 24:  # At least a day of hourly data
                logger.warning(f"Insufficient data for {symbol} after filtering")
                continue
                
            asset_data[symbol] = df
            logger.info(f"Loaded filtered data for {symbol}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {symbol}: {str(e)}")
            continue
            
    return asset_data


if __name__ == "__main__":
    # Example usage
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    selector = AssetUniverseSelector(
        data_dir=data_dir,
        lookback_days=30,
        max_assets=5,
        min_volume_usd=1000000,
        min_market_cap_usd=10000000
    )
    
    selected = selector.select_assets()
    print(f"Selected assets: {selected['selected_assets']}")
    
    # Load data for just the selected assets
    asset_data = load_and_filter_data(
        data_dir=data_dir,
        symbols=selected['selected_assets'],
        start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        end_date=None  # Up to the most recent available
    )
    
    print(f"Loaded data for {len(asset_data)} selected assets") 