# AI Trading System

A full-stack AI-based cryptocurrency trading system that uses reinforcement learning and multi-agent architecture for intelligent asset selection, trading, and portfolio management.

## Features

- **Asset Universe Selection**: Automatically selects the best assets to trade based on market data, volatility, volume, and other factors
- **RL-Based Trading Agents**: Individual PPO agents per asset for optimal trading decisions
- **Multi-GPU Training**: Scale agent training across multiple GPUs for maximum efficiency
- **Meta-Agent Portfolio Allocation**: AI-based portfolio optimization using techniques like risk parity and Markowitz optimization
- **Comprehensive Backtesting**: Test strategies against historical data with detailed performance metrics
- **LLM-Powered Reasoning**: Generate trade explanations and market analysis using local LLMs
- **Prometheus/Grafana Monitoring**: Track system performance and trading metrics in real-time
- **Modular Architecture**: Easy to extend and customize for different strategies and markets

## System Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│ Data Collection │────>│ Asset Universe    │────>│ Multi-Agent        │
│ & Processing    │     │ Selection (AI)    │     │ Training System    │
└─────────────────┘     └───────────────────┘     └────────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌───────────────────┐     ┌────────────────────┐
│ Monitoring      │<────│ Trading Execution │<────│ Meta-Agent         │
│ & Reporting     │     │ & Backtesting     │<────│ Portfolio Allocator│
└─────────────────┘     └───────────────────┘     └────────────────────┘
```

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU(s) for optimal performance (optional but recommended)
- 16GB+ RAM
- Historical market data (OHLCV) in CSV format

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd ai_trading_system
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install additional dependencies for Monitoring (optional):

```bash
pip install prometheus_client pandas psutil GPUtil
```

5. Install additional dependencies for LLM reasoning (optional):

```bash
pip install ollama requests
```

## Data Requirements

Place historical OHLCV data in the `data/` directory with the following format:
- Filename: `{symbol}_1h.csv` (e.g., `btc_usd_1h.csv`)
- Required columns: `datetime`, `open`, `high`, `low`, `close`, `volume`
- Datetime should be in a format parsable by pandas

Example data structure:
```
datetime,close,high,low,open,volume
2022-01-01 00:00:00,46209.25,46284.39,46192.52,46268.85,478.23
2022-01-01 01:00:00,46178.62,46241.41,46103.71,46209.25,512.67
...
```

## Configuration

The system uses a JSON configuration file (`config.json`) that can be customized:

```json
{
  "system": {
    "name": "AI Trading System",
    "version": "1.0.0",
    "mode": "backtest",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01"
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
    "allocation_method": "risk_parity",
    "max_allocation": 0.4,
    "min_allocation": 0.05,
    "risk_aversion": 1.2
  }
}
```

## Usage

### Running the Full Pipeline

To run the complete trading system pipeline:

```bash
python run_system.py --pipeline
```

This will:
1. Select the best assets to trade
2. Load historical data
3. Train RL agents for each asset
4. Run backtests to evaluate performance
5. Generate optimal portfolio allocation
6. Create a trading report

### Running Individual Components

You can also run individual components of the system:

```bash
# Select assets only
python run_system.py --select-assets

# Train agents only
python run_system.py --train

# Run backtest only
python run_system.py --backtest

# Generate portfolio allocation only
python run_system.py --allocate
```

### Multi-GPU Training

To train agents across multiple GPUs:

```bash
python agents/multi_gpu_trainer.py --assets-file outputs/top_assets.json --data-path data --output-path models --gpu-ids 0,1
```

### Backtesting

To run a standalone backtest:

```bash
python backtesting/backtest_engine.py --data-dir data --models-dir models --output-dir outputs --start-date 2022-01-01 --end-date 2023-01-01
```

## Output Files

The system generates various output files:

- `outputs/top_assets.json`: Selected assets with metrics
- `outputs/backtest_results.json`: Detailed backtest results
- `outputs/backtest_results.png`: Visual performance chart
- `outputs/portfolio_allocation.json`: Portfolio weights
- `outputs/trading_report.md`: LLM-generated trading report
- `logs/metrics.json`: Performance metrics history

## Monitoring

### Basic Metrics

Basic metrics are continuously logged to `logs/metrics.json` and can be exported to CSV:

```python
from monitoring.metrics_logger import MetricsLogger

logger = MetricsLogger()
metrics_csv = logger.export_metrics_csv(output_dir="reports")
```

### Prometheus/Grafana Setup

1. Install Prometheus:
   - Download from https://prometheus.io/download/
   - Configure prometheus.yml to scrape metrics from the system

2. Install Grafana:
   - Download from https://grafana.com/grafana/download
   - Configure datasource to use Prometheus
   - Import the provided dashboard template from `monitoring/grafana_dashboard.json`

3. Enable Prometheus in the system:
   - Set `enable_prometheus: True` in the MetricsLogger initialization

## Future Extensions

### Live Trading

To connect to exchanges for live trading:

1. Install CCXT:
   ```bash
   pip install ccxt
   ```

2. Configure exchange API credentials in `config.json`:
   ```json
   "exchange": {
     "name": "binance",
     "api_key": "YOUR_API_KEY",
     "api_secret": "YOUR_API_SECRET",
     "testnet": true
   }
   ```

3. Run with live trading mode:
   ```bash
   python run_system.py --pipeline --mode live
   ```

### Local LLM Setup

For local reasoning using Ollama:

1. Install Ollama:
   - Follow instructions at https://ollama.ai/

2. Pull the desired model:
   ```bash
   ollama pull llama2
   ```

3. Start Ollama server:
   ```bash
   ollama serve
   ```

4. Configure the reasoning engine to use the local model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stable-Baselines3 for RL implementations
- OpenAI for LLM reasoning capabilities
- Various academic papers on portfolio optimization and RL for trading 