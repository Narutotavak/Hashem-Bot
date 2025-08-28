# FreqAI Production-Ready Implementation

This repository contains a comprehensive, production-ready implementation of FreqAI (Freqtrade Artificial Intelligence) for cryptocurrency trading. The implementation follows the official Freqtrade documentation and provides a complete solution for machine learning-based trading strategies.

## 🚀 Features

- **Advanced Feature Engineering**: Comprehensive technical indicators with automatic expansion across timeframes, periods, and correlation pairs
- **Multiple Target Types**: Regression and classification targets for price movement, volatility, and trend prediction
- **Production-Ready Strategy**: Complete trading strategy with entry/exit conditions, trade confirmation, and custom stoploss
- **Outlier Detection**: Multiple outlier detection methods (DI, SVM, DBSCAN) with configurable thresholds
- **Comprehensive Testing**: Full test suite with unit tests, integration tests, and CI/CD pipeline
- **Professional Configuration**: Production-ready configuration with detailed parameter explanations

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Feature Engineering](#feature-engineering)
- [Strategy Details](#strategy-details)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

- Python 3.9+ (recommended: 3.11)
- Freqtrade with ML dependencies
- Sufficient disk space for model storage

### Install Freqtrade with ML Support

```bash
# Clone Freqtrade repository
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade

# Install with ML dependencies
./setup.sh -i

# When prompted, answer 'y' to install ML dependencies
# This will install: freqtrade[ml], torch, scikit-learn, etc.
```

### Alternative: Docker Installation

```bash
# Pull the FreqAI-enabled Docker image
docker pull freqtrade/freqtrade:develop_freqai

# Or use the provided docker-compose file
docker-compose -f docker/docker-compose-freqai.yml up -d
```

### Verify Installation

```bash
# Check if FreqAI is available
freqtrade --help | grep -i freqai

# Should show FreqAI-related commands
```

## ⚙️ Configuration

### Basic Configuration

The main configuration file is located at `config_examples/config_freqai.example.json`. This file contains all necessary parameters for a production FreqAI setup.

#### Key Configuration Sections

```json
{
  "freqai": {
    "enabled": true,
    "purge_old_models": 3,
    "train_period_days": 30,
    "backtest_period_days": 7,
    "identifier": "production-freqai-v1",
    "feature_parameters": {
      "include_timeframes": ["5m", "15m", "1h", "4h"],
      "include_corr_pairlist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
      "label_period_candles": 24,
      "include_shifted_candles": 3,
      "indicator_periods_candles": [10, 20, 50]
    }
  }
}
```

#### Parameter Explanations

| Parameter | Description | Default | When to Change |
|-----------|-------------|---------|----------------|
| `enabled` | Enable/disable FreqAI | `true` | Never change |
| `purge_old_models` | Number of models to keep | `3` | Based on disk space |
| `train_period_days` | Training data window | `30` | Market volatility |
| `backtest_period_days` | Backtest window | `7` | Retraining frequency |
| `identifier` | Unique model identifier | `production-freqai-v1` | New features/config |
| `label_period_candles` | Prediction horizon | `24` | Trading strategy |
| `include_timeframes` | Additional timeframes | `["5m", "15m", "1h", "4h"]` | Market analysis |
| `include_corr_pairlist` | Correlation pairs | `["BTC/USDT:USDT"]` | Market relationships |
| `indicator_periods_candles` | Technical indicator periods | `[10, 20, 50]` | Market cycles |

### Advanced Configuration

#### Outlier Detection

```json
{
  "freqai": {
    "feature_parameters": {
      "DI_threshold": 0.9,
      "use_SVM_to_remove_outliers": true,
      "use_DBSCAN_to_remove_outliers": false,
      "outlier_protection_percentage": 30
    }
  }
}
```

#### Model Training Parameters

```json
{
  "freqai": {
    "model_training_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.1,
      "max_depth": 6,
      "subsample": 0.8,
      "colsample_bytree": 0.8
    }
  }
}
```

## 🚀 Usage

### Training Models

#### Initial Training

```bash
# Train a new model with the example strategy
freqtrade trade \
  --strategy FreqaiExampleStrategy \
  --config config_examples/config_freqai.example.json \
  --freqaimodel LightGBMRegressor \
  --strategy-path freqtrade/templates
```

#### Training with Different Models

```bash
# CatBoost Regressor
freqtrade trade \
  --strategy FreqaiExampleStrategy \
  --config config_examples/config_freqai.example.json \
  --freqaimodel CatboostRegressor

# XGBoost Regressor
freqtrade trade \
  --strategy FreqaiExampleStrategy \
  --config config_examples/config_freqai.example.json \
  --freqaimodel XGBoostRegressor

# PyTorch MLP Regressor
freqtrade trade \
  --strategy FreqaiExampleStrategy \
  --config config_examples/config_freqai.example.json \
  --freqaimodel PyTorchMLPRegressor
```

### Backtesting

```bash
# Run backtesting with FreqAI
freqtrade backtesting \
  --strategy FreqaiExampleStrategy \
  --strategy-path freqtrade/templates \
  --config config_examples/config_freqai.example.json \
  --freqaimodel LightGBMRegressor \
  --timerange 20230101-20231201
```

### Live Trading

```bash
# Start live trading (dry-run first!)
freqtrade trade \
  --strategy FreqaiExampleStrategy \
  --config config_examples/config_freqai.example.json \
  --freqaimodel LightGBMRegressor \
  --strategy-path freqtrade/templates
```

### Hyperopt

```bash
# Optimize strategy parameters
freqtrade hyperopt \
  --hyperopt-loss SharpeHyperOptLoss \
  --strategy FreqaiExampleStrategy \
  --freqaimodel LightGBMRegressor \
  --strategy-path freqtrade/templates \
  --config config_examples/config_freqai.example.json \
  --timerange 20230101-20230601
```

## 🔧 Feature Engineering

### Feature Types

The strategy implements three types of feature engineering functions:

#### 1. `feature_engineering_expand_all()`

Features that expand across:
- Indicator periods: `[10, 20, 50]`
- Timeframes: `["5m", "15m", "1h", "4h"]`
- Shifted candles: `3`
- Correlation pairs: `["BTC/USDT:USDT", "ETH/USDT:USDT"]`

**Total features per indicator**: 19 × 3 × 4 × 3 × 3 = 2,052

**Features included**:
- Momentum: RSI, MFI, ADX, ROC, Williams %R
- Moving Averages: SMA, EMA, WMA
- Bollinger Bands: Width, Position, Squeeze
- Volume: Volume ratios, Volume momentum
- Price: Price changes, Volatility
- Oscillators: Stochastic, MACD

#### 2. `feature_engineering_expand_basic()`

Features that expand across timeframes and pairs but NOT periods:
- **Total features**: 16 × 4 × 3 × 3 = 576

**Features included**:
- Raw price/volume data
- Percentage changes
- Long-term moving averages (200, 50 periods)
- Volume analysis
- Price position relative to averages

#### 3. `feature_engineering_standard()`

Features that don't expand (called once with base timeframe):
- **Total features**: 16

**Features included**:
- Time-based: Day of week, Hour of day, Minute of hour
- Market sessions: London, NY, Asia, Weekend
- Candle patterns: Doji, Hammer, Engulfing, Morning/Evening Star
- Market structure: Higher highs, Lower lows
- Volatility: ATR, ATR ratio

### Target Generation

The strategy generates 5 targets:

1. **Price Movement** (`&-s_close`): Predicted price change over next N candles
2. **Volatility** (`&-s_volatility`): Predicted volatility over next N candles
3. **Trend Direction** (`&-s_trend`): Classification: "up", "down", "sideways"
4. **Maximum Drawdown** (`&-s_max_drawdown`): Predicted maximum drawdown
5. **Volume Ratio** (`&-s_volume_ratio`): Predicted volume ratio

## 📊 Strategy Details

### Entry Conditions

#### Long Entry
- `do_predict == 1` (prediction reliable)
- `&-s_close > 0.01` (predicted 1%+ increase)
- `&-s_trend == "up"` (trend prediction up)
- `&-s_volatility < 0.05` (low volatility)
- `DI_values < 1.5` (low dissimilarity)
- Z-score > 1.0 (statistical significance)

#### Short Entry
- `do_predict == 1` (prediction reliable)
- `&-s_close < -0.01` (predicted 1%+ decrease)
- `&-s_trend == "down"` (trend prediction down)
- `&-s_volatility < 0.05` (low volatility)
- `DI_values < 1.5` (low dissimilarity)

### Exit Conditions

#### Long Exit
- `do_predict == 1` (prediction reliable)
- `&-s_close < 0` (predicted decrease)
- `&-s_trend == "down"` (trend prediction down)

#### Short Exit
- `do_predict == 1` (prediction reliable)
- `&-s_close > 0` (predicted increase)
- `&-s_trend == "up"` (trend prediction up)

### Trade Confirmation

Additional checks before entering trades:
- Prediction confidence (`do_predict == 1`)
- DI values below threshold (`DI_values < 2.0`)
- Model not expired (`do_predict != 2`)
- Price slippage protection (0.25% max)

### Custom Stoploss

Dynamic stoploss based on volatility predictions:
- Normal volatility: Standard stoploss (-5%)
- High volatility (>5%): Adjusted stoploss (-7.5%)

## 🧪 Testing

### Running Tests

```bash
# Run all FreqAI tests
pytest tests/test_freqai_*.py -v

# Run specific test files
pytest tests/test_freqai_feature_engineering.py -v
pytest tests/test_freqai_strategy.py -v
pytest tests/test_freqai_configuration.py -v

# Run with coverage
pytest tests/test_freqai_*.py -v --cov=freqtrade.templates.FreqaiExampleStrategy --cov-report=html
```

### Test Coverage

The test suite covers:
- ✅ Feature engineering functions
- ✅ Target generation
- ✅ Entry/exit conditions
- ✅ Trade confirmation
- ✅ Custom stoploss
- ✅ Configuration validation
- ✅ Strategy integration
- ✅ Error handling

### CI/CD Pipeline

Automated testing via GitHub Actions:
- **Python versions**: 3.9, 3.10, 3.11
- **Code quality**: flake8, black, isort, mypy
- **Integration tests**: Configuration validation, strategy import
- **Documentation**: Docstring validation, configuration documentation

## 🔍 Troubleshooting

### Common Issues

#### 1. Model Training Fails

**Symptoms**: Training errors, NaN values in features
**Solutions**:
- Increase `startup_candle_count` (recommended: 2x max period)
- Check for sufficient historical data
- Verify indicator parameters are reasonable

```json
{
  "freqai": {
    "feature_parameters": {
      "indicator_periods_candles": [10, 20, 50]
    }
  }
}
```

**Recommended startup_candle_count**: 100 (2 × 50)

#### 2. Low Prediction Confidence

**Symptoms**: `do_predict != 1`, frequent trade rejections
**Solutions**:
- Lower `DI_threshold` (try 0.7-0.8)
- Disable aggressive outlier detection
- Increase training data period

```json
{
  "freqai": {
    "feature_parameters": {
      "DI_threshold": 0.7,
      "use_SVM_to_remove_outliers": false,
      "use_DBSCAN_to_remove_outliers": false
    }
  }
}
```

#### 3. Model Expiration

**Symptoms**: `do_predict == 2`, "model expired" warnings
**Solutions**:
- Increase `expiration_hours`
- Reduce `live_retrain_hours`
- Check training performance

```json
{
  "freqai": {
    "expiration_hours": 4,
    "live_retrain_hours": 2
  }
}
```

#### 4. High DI Values

**Symptoms**: High `DI_values`, unreliable predictions
**Solutions**:
- Increase training data diversity
- Add more correlation pairs
- Adjust feature engineering
- Check market conditions

#### 5. Feature Explosion

**Symptoms**: Too many features, slow training
**Solutions**:
- Reduce `include_timeframes`
- Reduce `include_shifted_candles`
- Enable PCA
- Use fewer indicator periods

```json
{
  "freqai": {
    "feature_parameters": {
      "principal_component_analysis": true,
      "include_timeframes": ["5m", "1h"],
      "include_shifted_candles": 1
    }
  }
}
```

### Performance Optimization

#### Memory Usage
```json
{
  "freqai": {
    "reduce_df_footprint": true,
    "data_kitchen_thread_count": 2
  }
}
```

#### Training Speed
```json
{
  "freqai": {
    "feature_parameters": {
      "principal_component_analysis": true,
      "plot_feature_importances": 0
    },
    "model_training_parameters": {
      "n_jobs": -1,
      "verbose": -1
    }
  }
}
```

## 📚 API Reference

### Strategy Methods

#### `feature_engineering_expand_all(dataframe, period, metadata)`
Creates features that expand across all dimensions.

**Parameters**:
- `dataframe`: OHLCV data
- `period`: Current indicator period
- `metadata`: Pair and timeframe information

**Returns**: DataFrame with added features

#### `feature_engineering_expand_basic(dataframe, metadata)`
Creates features that expand across timeframes and pairs.

**Parameters**:
- `dataframe`: OHLCV data
- `metadata`: Pair and timeframe information

**Returns**: DataFrame with added features

#### `feature_engineering_standard(dataframe, metadata)`
Creates features that don't expand (time-based, pattern-based).

**Parameters**:
- `dataframe`: OHLCV data
- `metadata`: Pair information

**Returns**: DataFrame with added features

#### `set_freqai_targets(dataframe, metadata)`
Creates prediction targets for the model.

**Parameters**:
- `dataframe`: OHLCV data
- `metadata`: Pair information

**Returns**: DataFrame with added targets

#### `populate_indicators(dataframe, metadata)`
Main indicator population function that calls FreqAI.

**Parameters**:
- `dataframe`: OHLCV data
- `metadata`: Pair information

**Returns**: DataFrame with indicators and predictions

#### `populate_entry_trend(dataframe, metadata)`
Defines entry conditions based on FreqAI predictions.

**Parameters**:
- `dataframe`: OHLCV data with predictions
- `metadata`: Pair information

**Returns**: DataFrame with entry signals

#### `populate_exit_trend(dataframe, metadata)`
Defines exit conditions based on FreqAI predictions.

**Parameters**:
- `dataframe`: OHLCV data with predictions
- `metadata`: Pair information

**Returns**: DataFrame with exit signals

#### `confirm_trade_entry(...)`
Additional confirmation before entering trades.

**Parameters**:
- `pair`: Trading pair
- `order_type`: Order type
- `amount`: Trade amount
- `rate`: Entry rate
- `side`: Long/short

**Returns**: Boolean confirmation

#### `custom_stoploss(...)`
Dynamic stoploss based on volatility predictions.

**Parameters**:
- `pair`: Trading pair
- `trade`: Trade object
- `current_rate`: Current price
- `current_profit`: Current profit

**Returns**: Stoploss percentage

### Configuration Parameters

#### Required Parameters
- `enabled`: Enable FreqAI
- `identifier`: Unique model identifier
- `train_period_days`: Training data window
- `backtest_period_days`: Backtest window

#### Feature Parameters
- `include_timeframes`: Additional timeframes
- `include_corr_pairlist`: Correlation pairs
- `label_period_candles`: Prediction horizon
- `indicator_periods_candles`: Technical indicator periods
- `include_shifted_candles`: Historical candle inclusion

#### Outlier Detection
- `DI_threshold`: Dissimilarity Index threshold
- `use_SVM_to_remove_outliers`: SVM outlier detection
- `use_DBSCAN_to_remove_outliers`: DBSCAN outlier detection
- `outlier_protection_percentage`: Maximum outlier percentage

#### Advanced Features
- `weight_factor`: Temporal weighting factor
- `principal_component_analysis`: Dimensionality reduction
- `noise_standard_deviation`: Training data noise
- `reverse_train_test_order`: Reverse train/test split

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions
- Keep functions focused and small
- Use meaningful variable names

### Testing Guidelines

- Write tests for all new functionality
- Maintain test coverage above 90%
- Use descriptive test names
- Mock external dependencies
- Test edge cases and error conditions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Freqtrade team for the excellent framework
- FreqAI contributors for the machine learning integration
- Technical analysis library maintainers
- Open source community

## 📞 Support

### Getting Help

1. **Documentation**: Check this README and official Freqtrade docs
2. **Issues**: Search existing issues or create new ones
3. **Discussions**: Use GitHub Discussions for questions
4. **Community**: Join Freqtrade Discord/Telegram

### Useful Links

- [Freqtrade Documentation](https://www.freqtrade.io/)
- [FreqAI Configuration](https://www.freqtrade.io/en/latest/freqai-configuration/)
- [FreqAI Parameter Table](https://www.freqtrade.io/en/latest/freqai-parameter-table/)
- [FreqAI Feature Engineering](https://www.freqtrade.io/en/latest/freqai-feature-engineering/)
- [FreqAI Running Guide](https://www.freqtrade.io/en/latest/freqai-running/)

---

**Note**: This implementation is for educational and research purposes. Always test thoroughly in dry-run mode before live trading. Cryptocurrency trading involves significant risk.
