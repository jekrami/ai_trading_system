# AI Trading System

A full-stack AI-based cryptocurrency trading system that uses reinforcement learning and multi-agent architecture for intelligent asset selection, trading, and portfolio management. **Now optimized for RTX 3090 with 3-5x faster training!**

## 🚀 Key Features

- **🎯 Real-Time Trading Signals**: Get specific BUY/SELL/HOLD recommendations with exact amounts and timing
- **⚡ RTX 3090 Optimized Training**: Native PyTorch implementation with mixed precision for maximum GPU utilization
- **🤖 Asset Universe Selection**: Automatically selects the best assets to trade based on market data, volatility, volume, and other factors
- **🧠 RL-Based Trading Agents**: Individual PPO agents per asset for optimal trading decisions
- **🔥 High-Performance Training**: 3-5x faster training with optimized batch sizes and neural networks
- **📊 Meta-Agent Portfolio Allocation**: AI-based portfolio optimization using techniques like risk parity and Markowitz optimization
- **📈 Comprehensive Backtesting**: Test strategies against historical data with detailed performance metrics
- **💬 LLM-Powered Reasoning**: Generate trade explanations and market analysis using local LLMs
- **📊 Prometheus/Grafana Monitoring**: Track system performance and trading metrics in real-time
- **🔧 Modular Architecture**: Easy to extend and customize for different strategies and markets

## 🎯 What You Get

The system provides **clear, actionable trading signals**:

```
🟢 BUY ETH_USD: 0.386448 units ($1,000) at $2,587.67 - Confidence: 85%
🟡 HOLD BTC_USD at $103,780.71 - Confidence: 60%
🔴 SELL SOL_USD: 0.289 units ($500) at $172.51 - Confidence: 78%
```

Each signal includes:
- **Action**: BUY/SELL/HOLD
- **Exact amount**: Position size in units and USD
- **Current price**: Real-time market price
- **Confidence level**: AI model confidence
- **Reasoning**: Why the AI made this decision

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
- **RTX 3090 or similar GPU** for optimal performance (24GB VRAM recommended)
- CUDA-compatible GPU(s) for training acceleration
- 16GB+ RAM (32GB+ recommended for large datasets)
- Historical market data (OHLCV) in CSV format

### 🚀 RTX 3090 Performance

With RTX 3090 optimizations:
- **Training Speed**: 3-5x faster than standard implementation
- **Batch Size**: 2048 (vs 64 standard)
- **Network Size**: 256x256x128 (vs 64x64 standard)
- **Memory Usage**: Optimized for 24GB VRAM
- **Mixed Precision**: FP16/FP32 for maximum speed

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

Place historical OHLCV data in the `data/` directory. The system supports multiple formats:

### Supported File Locations:
- `data/{symbol}_1h.csv` (e.g., `data/btc_usd_1h.csv`)
- `data/1h/{symbol}_1h.csv` (e.g., `data/1h/BTC_USD_1h.csv`)
- `data/{symbol}.csv` (e.g., `data/BTC_USD.csv`)

### Required Columns:
- `datetime`, `open`, `high`, `low`, `close`, `volume`
- Datetime should be in a format parsable by pandas

### Example Data Structure:
```csv
datetime,close,high,low,open,volume
2022-01-01 00:00:00,46209.25,46284.39,46192.52,46268.85,478.23
2022-01-01 01:00:00,46178.62,46241.41,46103.71,46209.25,512.67
...
```

### Sample Assets:
The system works with any cryptocurrency pairs like:
- BTC_USD, ETH_USD, SOL_USD, LINK_USD, XRP_USD
- ADA_USD, AVAX_USD, DOT_USD, MATIC_USD, DOGE_USD

## Configuration

The system uses a JSON configuration file (`config.json`) that can be customized:

### 🚀 RTX 3090 Optimized Configuration:

```json
{
  "system": {
    "name": "AI Trading System",
    "version": "1.0.0",
    "mode": "backtest",
    "start_date": "2024-05-17",
    "end_date": "2025-05-17"
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
    "batch_size": 2048,              // 🚀 RTX 3090 optimized
    "n_steps": 16384,                // 🚀 Large experience buffer
    "n_epochs": 10,                  // 🚀 Optimized epochs
    "learning_rate": 0.0003,
    "net_arch": [256, 256, 128],     // 🚀 Larger neural network
    "use_mixed_precision": true,     // 🚀 RTX 3090 acceleration
    "gradient_accumulation_steps": 4, // 🚀 Better gradients
    "use_native_pytorch": true       // 🚀 Maximum GPU utilization
  },
  "portfolio": {
    "allocation_method": "risk_parity",
    "max_allocation": 0.4,
    "min_allocation": 0.05,
    "risk_aversion": 1.2
  }
}
```

### Key RTX 3090 Optimizations:
- **batch_size: 2048** - 32x larger than standard (64)
- **net_arch: [256, 256, 128]** - 4x larger network
- **use_native_pytorch: true** - Maximum GPU utilization
- **use_mixed_precision: true** - FP16/FP32 for speed

## Usage

### 🚀 Quick Start (RTX 3090 Optimized)

#### **Option 1: Full Pipeline (Recommended)**
```bash
python run_system.py --pipeline
```

This will:
1. ✅ Select the best assets to trade
2. ✅ Load historical data
3. ✅ **Train RL agents with RTX 3090 optimization** (3-5x faster!)
4. ✅ **Generate real-time trading signals** (NEW!)
5. ✅ Run backtests to evaluate performance
6. ✅ Generate optimal portfolio allocation
7. ✅ Create a comprehensive trading report

#### **Option 2: Quick Trading Signals**
```bash
# Generate signals with simple momentum strategy
python scripts/generate_signals.py --balance 10000

# Generate signals from trained AI models
python run_system.py --signals
```

#### **Option 3: RTX 3090 Optimized Training**
```bash
# Single asset with maximum GPU utilization
python scripts/quick_train_rtx3090.py --symbol BTC_USD --timesteps 100000 --native

# Multiple assets
python scripts/quick_train_rtx3090.py --symbols BTC_USD ETH_USD SOL_USD --native
```

### Running Individual Components

```bash
# Select assets only
python run_system.py --select-assets

# Train agents with RTX 3090 optimization
python run_system.py --train

# Generate trading signals
python run_system.py --signals

# Run backtest only
python run_system.py --backtest

# Generate portfolio allocation only
python run_system.py --allocate
```

### 🔧 GPU Optimization Tools

```bash
# Benchmark your GPU for optimal settings
python scripts/gpu_benchmark.py --benchmark

# Monitor GPU during training
python scripts/gpu_benchmark.py --monitor 5
```

### Advanced Usage

#### **Multi-GPU Training (Legacy)**
```bash
python agents/multi_gpu_trainer.py --assets-file outputs/top_assets.json --data-path data --output-path models --gpu-ids 0,1
```

#### **Standalone Backtesting**
```bash
python backtesting/backtest_engine.py --data-dir data --models-dir models --output-dir outputs --start-date 2022-01-01 --end-date 2023-01-01
```

## 📁 Output Files

The system generates comprehensive output files:

### 🎯 Trading Signals
- `outputs/trading_signals.json`: **Real-time BUY/SELL/HOLD signals with amounts**
- Console output: Human-readable trading recommendations

### 📊 Analysis & Reports
- `outputs/top_assets.json`: Selected assets with metrics
- `outputs/backtest_results.json`: Detailed backtest results
- `outputs/backtest_results.png`: Visual performance chart
- `outputs/portfolio_allocation.json`: Portfolio weights
- `outputs/trading_report.md`: LLM-generated trading report

### 🤖 AI Models
- `models/native_ppo_{symbol}.pt`: RTX 3090 optimized PyTorch models
- `models/ppo_{symbol}.zip`: Standard Stable-Baselines3 models

### 📈 Performance Data
- `logs/metrics.json`: Performance metrics history
- `gpu_monitoring.png`: GPU utilization charts

## 🎯 Example Trading Signal Output

```
================================================================================
🚀 AI TRADING SIGNALS - WHAT TO BUY/SELL/HOLD
================================================================================
📅 Generated: 2025-09-04 19:51:05
💰 Portfolio Balance: $10,000.00
📊 Total Assets: 5

🟢 BUY  ETH_USD
   💵 Current Price: $2,587.67
   📈 24h Change: +3.53%
   📏 Recommended Size: 0.386448 units
   💲 Trade Value: $1,000.00
   🎯 Confidence: 85.3%
   💭 Reasoning: Strong upward momentum suggests buying opportunity

🟡 HOLD BTC_USD
   💵 Current Price: $103,780.71
   📈 24h Change: +1.80%
   🎯 Confidence: 60.0%
   💭 Reasoning: Sideways movement suggests holding position

📈 TRADING SUMMARY:
   🟢 Buy Signals: 1
   🔴 Sell Signals: 0
   🟡 Hold Signals: 4
   💰 Total Trade Value: $1,000.00
================================================================================
```

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

## 🚀 Performance Benchmarks

### RTX 3090 Training Speed:
- **5,000 timesteps**: ~59 seconds (native PyTorch)
- **50,000 timesteps**: ~9.3 minutes (native PyTorch)
- **100,000 timesteps**: ~18.6 minutes (native PyTorch)
- **Speedup**: 3-5x faster than standard implementation

### Memory Utilization:
- **RTX 3090 VRAM**: 25.77GB total
- **Optimized Usage**: <1GB (plenty of headroom for larger models)
- **Batch Processing**: 2048 samples simultaneously

### Training Throughput:
- **Native PyTorch**: ~89 timesteps/second
- **Optimized SB3**: ~82 timesteps/second
- **Standard Setup**: ~30 timesteps/second

## 🔧 Troubleshooting

### Common Issues:

1. **"No asset data available"**
   - Ensure data files are in `data/` or `data/1h/` directories
   - Check file naming: `{SYMBOL}_1h.csv` or `{SYMBOL}.csv`

2. **"No trained agents available"**
   - Run training first: `python run_system.py --train`
   - Or use simple signals: `python scripts/generate_signals.py`

3. **GPU not detected**
   - Install CUDA-compatible PyTorch
   - Check GPU with: `python scripts/gpu_benchmark.py --benchmark`

4. **Out of memory errors**
   - Reduce batch_size in config.json
   - Use gradient_accumulation_steps for larger effective batches

## 📚 Additional Resources

- **RTX 3090 Optimization Guide**: `RTX3090_OPTIMIZATION_GUIDE.md`
- **GPU Benchmark Results**: Run `python scripts/gpu_benchmark.py --benchmark`
- **Training Logs**: Check `logs/` directory for detailed training information

## Acknowledgments

- Stable-Baselines3 for RL implementations
- PyTorch for native GPU optimization capabilities
- OpenAI for LLM reasoning capabilities
- NVIDIA for RTX 3090 hardware acceleration
- Various academic papers on portfolio optimization and RL for trading