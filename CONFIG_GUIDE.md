# Complete Configuration Guide for config.json

This guide explains every option in the `config.json` file with examples and recommendations for beginners.

## 📋 Table of Contents

1. [System Configuration](#system-configuration)
2. [Asset Selection](#asset-selection)
3. [Training Configuration](#training-configuration)
4. [Portfolio Management](#portfolio-management)
5. [Complete Example](#complete-example)
6. [Beginner Recommendations](#beginner-recommendations)

---

## 🖥️ System Configuration

```json
{
  "system": {
    "name": "AI Trading System",
    "version": "1.0.0",
    "mode": "backtest",
    "start_date": "2024-05-17",
    "end_date": "2025-05-17"
  }
}
```

### Options Explained:

| Option | Description | Values | Beginner Recommendation |
|--------|-------------|--------|-------------------------|
| `name` | System identifier | Any string | Keep default |
| `version` | Version tracking | Any string | Keep default |
| `mode` | Operation mode | `"backtest"`, `"live"`, `"paper"` | Use `"backtest"` |
| `start_date` | Data start date | `"YYYY-MM-DD"` format | Use recent 1-year period |
| `end_date` | Data end date | `"YYYY-MM-DD"` format | Use current date |

**💡 Beginner Tip:** Always use `"backtest"` mode for testing. Never use `"live"` until you're confident in your strategy.

---

## 🎯 Asset Selection

```json
{
  "asset_selection": {
    "max_assets": 5,
    "lookback_days": 30,
    "min_volume_usd": 1000000,
    "min_market_cap_usd": 10000000,
    "volatility_threshold": 0.02,
    "correlation_threshold": 0.8
  }
}
```

### Options Explained:

| Option | Description | Range | Beginner Recommendation |
|--------|-------------|-------|-------------------------|
| `max_assets` | Maximum number of assets to select | 1-20 | **5** (manageable portfolio) |
| `lookback_days` | Days to analyze for selection | 7-90 | **30** (good balance) |
| `min_volume_usd` | Minimum daily trading volume | 100K-10M+ | **1M** (ensures liquidity) |
| `min_market_cap_usd` | Minimum market capitalization | 1M-100B+ | **10M** (avoids micro-caps) |
| `volatility_threshold` | Minimum volatility for selection | 0.01-0.1 | **0.02** (2% daily volatility) |
| `correlation_threshold` | Maximum correlation between assets | 0.5-0.9 | **0.8** (reduces redundancy) |

**💡 Beginner Tips:**
- Start with 3-5 assets to keep it simple
- Higher `min_volume_usd` = more stable, liquid assets
- Lower `correlation_threshold` = more diversified portfolio

---

## 🚀 Training Configuration

```json
{
  "training": {
    "window_size": 24,
    "initial_balance": 10000.0,
    "transaction_fee": 0.001,
    "timesteps": 100000,
    "batch_size": 2048,
    "n_steps": 16384,
    "n_epochs": 10,
    "learning_rate": 0.0003,
    "net_arch": [256, 256, 128],
    "use_mixed_precision": true,
    "gradient_accumulation_steps": 4,
    "use_native_pytorch": true
  }
}
```

### Basic Options:

| Option | Description | Range | Beginner Recommendation |
|--------|-------------|-------|-------------------------|
| `window_size` | Hours of price history to analyze | 12-48 | **24** (1 day of data) |
| `initial_balance` | Starting portfolio value (USD) | 1K-100K+ | **10000** ($10K portfolio) |
| `transaction_fee` | Trading fee percentage | 0.0001-0.01 | **0.001** (0.1% fee) |
| `timesteps` | Training iterations per asset | 10K-1M+ | **100000** (good training) |

### Advanced RTX 3090 Options:

| Option | Description | Values | RTX 3090 Optimized |
|--------|-------------|--------|-------------------|
| `batch_size` | Samples processed simultaneously | 64-4096 | **2048** (optimal for 24GB) |
| `n_steps` | Experience buffer size | 1024-32768 | **16384** (8x batch_size) |
| `n_epochs` | Training epochs per update | 5-20 | **10** (good convergence) |
| `learning_rate` | AI learning speed | 0.0001-0.001 | **0.0003** (stable learning) |
| `net_arch` | Neural network layers | `[64,64]` to `[512,512,256]` | **[256,256,128]** (powerful) |
| `use_mixed_precision` | FP16/FP32 training | `true`/`false` | **true** (RTX 3090 speed) |
| `gradient_accumulation_steps` | Gradient batching | 1-8 | **4** (better gradients) |
| `use_native_pytorch` | Maximum GPU utilization | `true`/`false` | **true** (fastest training) |

**💡 Beginner Tips:**
- **Don't change** RTX 3090 optimized settings unless you know what you're doing
- Increase `timesteps` for better AI training (but longer time)
- Lower `learning_rate` if training is unstable

### Performance vs Quality Trade-offs:

| Setting | Fast Training | Balanced | High Quality |
|---------|---------------|----------|--------------|
| `timesteps` | 50,000 | 100,000 | 500,000 |
| `batch_size` | 1024 | 2048 | 4096 |
| `n_epochs` | 5 | 10 | 20 |
| Training Time | ~5 min/asset | ~10 min/asset | ~50 min/asset |

---

## 💼 Portfolio Management

```json
{
  "portfolio": {
    "allocation_method": "risk_parity",
    "max_allocation": 0.4,
    "min_allocation": 0.05,
    "risk_aversion": 1.2,
    "rebalance_frequency": "weekly",
    "stop_loss": 0.1,
    "take_profit": 0.2
  }
}
```

### Options Explained:

| Option | Description | Values | Beginner Recommendation |
|--------|-------------|--------|-------------------------|
| `allocation_method` | How to distribute money | `"equal"`, `"risk_parity"`, `"markowitz"` | **"risk_parity"** (balanced risk) |
| `max_allocation` | Maximum % per asset | 0.1-0.8 | **0.4** (40% max per asset) |
| `min_allocation` | Minimum % per asset | 0.01-0.2 | **0.05** (5% min per asset) |
| `risk_aversion` | Risk tolerance | 0.5-3.0 | **1.2** (moderate risk) |
| `rebalance_frequency` | How often to rebalance | `"daily"`, `"weekly"`, `"monthly"` | **"weekly"** (good balance) |
| `stop_loss` | Maximum loss before selling | 0.05-0.3 | **0.1** (10% stop loss) |
| `take_profit` | Profit target before selling | 0.1-0.5 | **0.2** (20% take profit) |

**💡 Allocation Methods Explained:**
- **`"equal"`**: Same amount in each asset (simple)
- **`"risk_parity"`**: Balance risk across assets (recommended)
- **`"markowitz"`**: Optimize risk/return mathematically (advanced)

---

## 📝 Complete Example Configuration

Here's a complete `config.json` with beginner-friendly settings:

```json
{
  "system": {
    "name": "My AI Trading System",
    "version": "1.0.0",
    "mode": "backtest",
    "start_date": "2024-01-01",
    "end_date": "2025-01-01"
  },
  "asset_selection": {
    "max_assets": 5,
    "lookback_days": 30,
    "min_volume_usd": 1000000,
    "min_market_cap_usd": 10000000,
    "volatility_threshold": 0.02,
    "correlation_threshold": 0.8
  },
  "training": {
    "window_size": 24,
    "initial_balance": 10000.0,
    "transaction_fee": 0.001,
    "timesteps": 100000,
    "batch_size": 2048,
    "n_steps": 16384,
    "n_epochs": 10,
    "learning_rate": 0.0003,
    "net_arch": [256, 256, 128],
    "use_mixed_precision": true,
    "gradient_accumulation_steps": 4,
    "use_native_pytorch": true
  },
  "portfolio": {
    "allocation_method": "risk_parity",
    "max_allocation": 0.4,
    "min_allocation": 0.05,
    "risk_aversion": 1.2,
    "rebalance_frequency": "weekly",
    "stop_loss": 0.1,
    "take_profit": 0.2
  }
}
```

---

## 🎓 Beginner Recommendations

### 🟢 Safe Starting Configuration:
```json
{
  "training": {
    "timesteps": 50000,        // Faster training for testing
    "initial_balance": 10000,  // $10K virtual portfolio
    "transaction_fee": 0.001   // 0.1% trading fee
  },
  "asset_selection": {
    "max_assets": 3,           // Start with just 3 assets
    "min_volume_usd": 5000000  // Only major cryptocurrencies
  },
  "portfolio": {
    "allocation_method": "equal", // Simple equal allocation
    "max_allocation": 0.5,        // Max 50% per asset
    "stop_loss": 0.15            // 15% stop loss (conservative)
  }
}
```

### 🟡 Intermediate Configuration:
```json
{
  "training": {
    "timesteps": 100000,       // Standard training
    "initial_balance": 25000   // Larger portfolio
  },
  "asset_selection": {
    "max_assets": 5,           // More diversification
    "min_volume_usd": 1000000  // Include more assets
  },
  "portfolio": {
    "allocation_method": "risk_parity", // Risk-balanced
    "stop_loss": 0.1,                  // 10% stop loss
    "take_profit": 0.2                 // 20% take profit
  }
}
```

### 🔴 Advanced Configuration:
```json
{
  "training": {
    "timesteps": 500000,       // Extensive training
    "batch_size": 4096,        // Larger batches
    "n_epochs": 20             // More training epochs
  },
  "asset_selection": {
    "max_assets": 10,          // Full diversification
    "correlation_threshold": 0.6 // Lower correlation
  },
  "portfolio": {
    "allocation_method": "markowitz", // Advanced optimization
    "risk_aversion": 0.8,            // Higher risk tolerance
    "rebalance_frequency": "daily"    // More frequent rebalancing
  }
}
```

---

## ⚠️ Common Mistakes to Avoid

1. **❌ Too many assets**: Don't start with more than 5 assets
2. **❌ Too low volume**: Avoid assets with < 1M daily volume
3. **❌ Too high learning rate**: Keep it at 0.0003 or lower
4. **❌ Too few timesteps**: Use at least 50,000 for decent training
5. **❌ No stop loss**: Always set a stop loss (10-15%)

---

## 🔧 Troubleshooting

### Training Too Slow?
- Reduce `timesteps` to 50,000
- Reduce `max_assets` to 3
- Keep RTX 3090 optimized settings

### Poor Performance?
- Increase `timesteps` to 200,000+
- Lower `learning_rate` to 0.0001
- Increase `window_size` to 48

### Out of Memory?
- Reduce `batch_size` to 1024
- Reduce `net_arch` to [128, 128]
- Set `use_mixed_precision` to false

---

**💡 Remember**: Start with conservative settings and gradually experiment as you gain experience!
